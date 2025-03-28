use crate::data_type::DataType;
use crate::definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK};
use crate::exerr;
use crate::result::to_ex;
use crate::{operators::Operator, ExError, ExResult};
use lazy_static::lazy_static;
use regex::Regex;
use smallvec::SmallVec;
use std::fmt::Debug;
use std::mem;

#[derive(Debug, PartialEq, Eq)]
pub enum Paren {
    Open,
    Close,
}

#[derive(PartialEq, Eq)]
pub enum ParsedToken<'a, T: DataType> {
    Num(T),
    Paren(Paren),
    /// index in the list of operators and operator
    Op((usize, Operator<'a, T>)),
    Var(&'a str),
}
impl<T> Debug for ParsedToken<'_, T>
where
    T: DataType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Num(x) => f.write_str(format!("{x:?}").as_str()),
            Self::Op(op) => f.write_str(op.1.repr()),
            Self::Paren(Paren::Open) => f.write_str("("),
            Self::Paren(Paren::Close) => f.write_str(")"),
            Self::Var(v) => f.write_str(v),
        }
    }
}

/// Returns the index of the variable in the slice. Panics if not available!
pub fn find_var_index(name: &str, parsed_vars: &[&str]) -> usize {
    let idx = parsed_vars.iter().enumerate().find(|(_, n)| **n == name);
    match idx {
        Some((i, _)) => i,
        None => {
            panic!("This is probably a bug. I don't know variable {name}")
        }
    }
}

/// Disambiguates operators based on predecessor token.
pub fn is_operator_binary<'a, T: DataType>(
    op: &Operator<'a, T>,
    parsed_token_on_the_left: Option<&ParsedToken<'a, T>>,
) -> ExResult<bool> {
    if op.has_bin() && !op.has_unary() {
        match parsed_token_on_the_left {
            Some(ParsedToken::Op(op_)) => Err(exerr!(
                "a binary operator cannot be on the right of another operator, {:?} next to {:?}",
                op,
                op_
            )),
            _ => Ok(true),
        }
    } else if op.has_bin() && op.has_unary() {
        Ok(match parsed_token_on_the_left {
            Some(ParsedToken::Num(_)) | Some(ParsedToken::Var(_)) => true,
            Some(ParsedToken::Paren(p)) => *p == Paren::Close,
            Some(ParsedToken::Op(_)) => false,
            None => false,
        })
    } else {
        Ok(false)
    }
}

/// Returns variable names in sorted order.
pub fn find_parsed_vars<'a, T: DataType>(
    parsed_tokens: &[ParsedToken<'a, T>],
) -> SmallVec<[&'a str; N_VARS_ON_STACK]> {
    let mut found_vars = SmallVec::<[&str; N_VARS_ON_STACK]>::new();
    for pt in parsed_tokens {
        match pt {
            ParsedToken::Var(name) if !found_vars.contains(name) => {
                found_vars.push(*name);
            }
            _ => (),
        }
    }
    found_vars.sort_unstable();
    found_vars
}

pub fn is_numeric_text(text: &str) -> Option<&str> {
    let mut n_dots = 0;
    let n_num_chars = text
        .chars()
        .take_while(|c| {
            let is_dot = *c == '.';
            if is_dot {
                n_dots += 1;
            }
            c.is_ascii_digit() || is_dot
        })
        .count();
    if (n_num_chars > 1 && n_dots < 2) || (n_num_chars == 1 && n_dots == 0) {
        Some(&text[0..n_num_chars])
    } else {
        None
    }
}

fn next_char_boundary(text: &str, start_idx: usize) -> usize {
    (1..text.len())
        .find(|idx| text.is_char_boundary(start_idx + idx))
        .expect("there has to be a char boundary somewhere")
}

fn find_op_of_comma<T>(parsed_tokens: &[ParsedToken<T>]) -> Option<usize>
where
    T: DataType,
{
    let paren_counter = parsed_tokens.iter().rev().scan(0, |state, pt| {
        *state += match pt {
            ParsedToken::Paren(Paren::Close) => -1,
            ParsedToken::Paren(Paren::Open) => 1,
            _ => 0,
        };
        Some(*state)
    });

    let rev_idx = parsed_tokens
        .iter()
        .rev()
        .zip(paren_counter)
        .enumerate()
        .find(|(_, (pt, paren_cnt))| {
            matches!(pt, 
            ParsedToken::Op(_) if *paren_cnt == 1)
        })
        .map(|(i, _)| i);

    rev_idx.map(|ridx| parsed_tokens.len() - 1 - ridx)
}

/// Parses tokens of a text with regexes and returns them as a vector
///
/// # Arguments
///
/// * `text` - text to be parsed
/// * `ops_in` - slice of operator-pairs
/// * `is_numeric` - closure that decides whether the current rest of the text starts with a number
///
/// # Errors
///
/// See [`parse_with_number_pattern`](parse_with_number_pattern)
///
pub fn tokenize_and_analyze<'a, T, F>(
    text: &'a str,
    ops_in: &[Operator<'a, T>],
    is_numeric: F,
) -> ExResult<SmallVec<[ParsedToken<'a, T>; N_NODES_ON_STACK]>>
where
    <T as std::str::FromStr>::Err: Debug,
    T: DataType,
    F: Fn(&'a str) -> Option<&'a str>,
{
    // We sort operators inverse alphabetically such that log2 has higher priority than log (wlog :D).
    let mut ops_tmp = ops_in
        .iter()
        .enumerate()
        .clone()
        .collect::<SmallVec<[_; 64]>>();
    ops_tmp.sort_unstable_by(|(_, o1), (_, o2)| o2.repr().partial_cmp(o1.repr()).unwrap());
    let ops_with_idx = ops_tmp; // from now on const

    lazy_static! {
        static ref RE_VAR_NAME: Regex =
            Regex::new(r"^[a-zA-Zα-ωΑ-Ω_]+[a-zA-Zα-ωΑ-Ω_0-9]*").unwrap();
    }
    lazy_static! {
        static ref RE_VAR_NAME_EXACT: Regex =
            Regex::new(r"^[a-zA-Zα-ωΑ-Ω_]+[a-zA-Zα-ωΑ-Ω_0-9]*$").unwrap();
    }

    let find_ops = |byte_offset: usize| {
        ops_with_idx.iter().find(|(_, op)| {
            let range_end = byte_offset + op.repr().len();
            if let Some(maybe_op) = text.get(byte_offset..range_end) {
                op.repr() == maybe_op
                    && (op.has_bin()
                        || range_end >= text.len()
                        || !RE_VAR_NAME_EXACT.is_match(
                            &text[byte_offset..range_end + next_char_boundary(text, range_end)],
                        ))
            } else {
                false
            }
        })
    };
    let mut res: SmallVec<[_; N_NODES_ON_STACK]> = SmallVec::new();
    let mut cur_byte_offset = 0usize;
    let mut close_additional_paren = false;
    let mut open_paren_count = 0;
    for (i, c) in text.char_indices() {
        if c == ' ' && i == cur_byte_offset {
            cur_byte_offset += 1;
        } else if i == cur_byte_offset && cur_byte_offset < text.len() {
            let text_rest = &text[cur_byte_offset..];
            let cur_byte_offset_tmp = cur_byte_offset;
            if c == '(' {
                cur_byte_offset += 1;
                res.push(ParsedToken::<T>::Paren(Paren::Open));
                open_paren_count += 1;
            } else if c == ')' {
                cur_byte_offset += 1;
                open_paren_count -= 1;
                res.push(ParsedToken::<T>::Paren(Paren::Close));
                if close_additional_paren && open_paren_count == 0 {
                    res.push(ParsedToken::Paren(Paren::Close));
                    close_additional_paren = false;
                }
            } else if c == ',' {
                // this is for binary operators with function call syntax.
                // we simply replace op(a,b) by ((a)op(b)) where the outer parens
                // are added to increase the priority as expected from the function
                // call syntax
                cur_byte_offset += 1;
                let op_idx = find_op_of_comma(&res).ok_or_else(|| {
                    exerr!("could not find operator for comma, could be operator with more than 2 args (not supported), missing operator, or paren mismatch",)
                })?;
                let op_at_comma = mem::replace(&mut res[op_idx], ParsedToken::Paren(Paren::Open));
                close_additional_paren = true;
                open_paren_count = 1;
                res.push(ParsedToken::Paren(Paren::Close));
                res.push(op_at_comma);
                res.push(ParsedToken::Paren(Paren::Open));
            } else if c == '{' {
                let n_count = text_rest
                    .chars()
                    .take_while(|c| *c != '}')
                    .map(|c| c.len_utf8())
                    .sum();
                let var_name = &text_rest[1..n_count];
                cur_byte_offset += n_count + 1;
                res.push(ParsedToken::Var(var_name));
                if close_additional_paren && open_paren_count == 0 {
                    res.push(ParsedToken::Paren(Paren::Close));
                    close_additional_paren = false;
                }
            } else if let Some(num_str) = is_numeric(text_rest) {
                let n_bytes = num_str.len();
                cur_byte_offset += n_bytes;
                res.push(ParsedToken::<T>::Num(num_str.parse::<T>().map_err(to_ex)?));
                if close_additional_paren && open_paren_count == 0 {
                    res.push(ParsedToken::Paren(Paren::Close));
                    close_additional_paren = false;
                }
            } else if let Some((idx, op)) = find_ops(cur_byte_offset_tmp) {
                let n_bytes = op.repr().len();
                cur_byte_offset += n_bytes;
                res.push(match op.constant() {
                    Some(constant) => ParsedToken::<T>::Num(constant),
                    None => ParsedToken::<T>::Op((*idx, (*op).clone())),
                });
            } else if let Some(var_str) = RE_VAR_NAME.find(text_rest) {
                let var_str = var_str.as_str();
                let n_bytes = var_str.len();
                cur_byte_offset += n_bytes;
                res.push(ParsedToken::<T>::Var(var_str));
                if close_additional_paren && open_paren_count == 0 {
                    res.push(ParsedToken::Paren(Paren::Close));
                    close_additional_paren = false;
                }
            } else {
                return Err(exerr!("don't know how to parse {}", text_rest));
            }
        }
    }
    Ok(res)
}

struct PairPreCondition<'a, T: DataType> {
    apply: fn(&ParsedToken<'a, T>, &ParsedToken<'a, T>) -> ExResult<()>,
}

fn make_err<T: DataType>(msg: &str, left: &ParsedToken<T>, right: &ParsedToken<T>) -> ExResult<()> {
    Err(exerr!("{}, left: {:?}; right: {:?}", msg, left, right))
}

fn make_pair_pre_conditions<'a, T: DataType>() -> [PairPreCondition<'a, T>; 7] {
    [
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Num(_))
                | (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Var(_))
                | (ParsedToken::Num(_), ParsedToken::Paren(_p @ Paren::Open))
                | (ParsedToken::Var(_), ParsedToken::Paren(_p @ Paren::Open)) => make_err(
                    "wlog a number/variable cannot be on the right of a closing parenthesis",
                    left,
                    right,
                ),
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Num(_), ParsedToken::Op((_, op)))
                | (ParsedToken::Var(_), ParsedToken::Op((_, op)))
                    // we do not ask for is_unary since operators can be both
                    if !op.has_bin() => make_err(
                        "a number/variable cannot be on the left of a unary operator",
                        left,
                        right,
                    ),
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                    (ParsedToken::Op((_, op_l)), ParsedToken::Op((_, op_r)))
                        if !op_l.has_unary() && !op_r.has_unary() => Err(exerr!(
                            "a binary operator cannot be next to the binary operator, violated by '{}' left of '{}'",
                            op_l.repr(),
                            op_r.repr())),
                    _ => Ok(()),
                }
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                    (ParsedToken::Op((_, op_l)), ParsedToken::Op((_, op_r)))
                        if !op_l.has_bin() && !op_r.has_unary() => Err(exerr!(
                            "a unary operator cannot be on the left of a binary one, violated by '{}' left of '{}'",
                            op_l.repr(),
                            op_r.repr())),
                    _ => Ok(()),
                }
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Op((_, op)), ParsedToken::Paren(_p @ Paren::Close)) => Err(exerr!(
                    "an operator cannot be on the left of a closing paren, violated by '{}'",
                    op.repr()
                )),
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                    (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Op((_, op))) if !op.has_bin() => {
                        Err(exerr!("a unary operator cannot be on the right of a closing paren, violated by '{}'", 
                            op.repr()))
                    }
                    _ => Ok(()),
                }
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (
                    ParsedToken::Paren(_p_l @ Paren::Open),
                    ParsedToken::Paren(_p_r @ Paren::Close),
                ) => make_err(
                    "wlog an opening paren cannot be next to a closing paren",
                    left,
                    right,
                ),
                _ => Ok(()),
            },
        },
    ]
}

/// Tries to give useful error messages for invalid constellations of the parsed tokens
///
/// # Arguments
///
/// * `parsed_tokens` - parsed tokens
///
/// # Errors
///
/// See [`parse_with_number_pattern`](parse_with_number_pattern)
///
pub fn check_parsed_token_preconditions<T>(parsed_tokens: &[ParsedToken<T>]) -> ExResult<()>
where
    T: DataType,
{
    if parsed_tokens.is_empty() {
        return Err(ExError::new("cannot parse empty string"));
    };

    let pair_pre_conditions = make_pair_pre_conditions::<T>();
    (0..parsed_tokens.len() - 1)
        .map(|i| {
            let failed = pair_pre_conditions
                .iter()
                .map(|ppc| (ppc.apply)(&parsed_tokens[i], &parsed_tokens[i + 1]))
                .find(|ppc_res| ppc_res.is_err());
            match failed {
                Some(failed_ppc) => failed_ppc,
                None => Ok(()),
            }
        })
        .collect::<ExResult<Vec<_>>>()?;

    let mut open_paren_cnt = 0i32;
    parsed_tokens
        .iter()
        .enumerate()
        .map(|(i, expr_elt)| -> ExResult<()> {
            match expr_elt {
                ParsedToken::Paren(p) => {
                    open_paren_cnt += match p {
                        Paren::Close => -1,
                        Paren::Open => 1,
                    };
                    if open_paren_cnt < 0 {
                        return Err(exerr!("too many closing parentheses until position {}", i));
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        })
        .collect::<ExResult<Vec<_>>>()?;
    if open_paren_cnt != 0 {
        Err(ExError::new("parentheses mismatch"))
    } else if let ParsedToken::Op(_) = parsed_tokens[parsed_tokens.len() - 1] {
        Err(ExError::new("the last element cannot be an operator"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
use crate::operators::{FloatOpsFactory, MakeOperators};

#[test]
fn test_is_numeric() {
    assert_eq!(is_numeric_text("5/6").unwrap(), "5");
    assert!(is_numeric_text(".").is_none());
    assert!(is_numeric_text("o.4").is_none());
    assert_eq!(is_numeric_text("6").unwrap(), "6");
    assert_eq!(is_numeric_text("4.").unwrap(), "4.");
    assert_eq!(is_numeric_text(".4").unwrap(), ".4");
    assert_eq!(is_numeric_text("23.414").unwrap(), "23.414");
}

#[test]
fn test_preconditions() {
    fn test(text: &str, msg_part: &str) {
        fn check_err_msg<X>(err: ExResult<X>, msg_part: &str) {
            match err {
                Ok(_) => {
                    println!("expected an error that should contain '{}'", msg_part);
                    unreachable!();
                }
                Err(e) => {
                    println!("msg '{}' should contain '{}'", e.msg(), msg_part);
                    assert!(e.msg().contains(msg_part));
                }
            }
        }

        let ops = FloatOpsFactory::<f32>::make();
        let elts = tokenize_and_analyze(text, &ops, is_numeric_text);
        println!("{:?}", elts);
        match elts {
            Err(e) => check_err_msg::<Vec<ParsedToken<f32>>>(Err(e), msg_part),
            Ok(elts) => {
                let error = check_parsed_token_preconditions(&elts);
                check_err_msg(error, msg_part);
            }
        };
    }
    test(
        r"3 * log2 * 5",
        r"a unary operator cannot be on the left of a binary",
    );
    test("xo-17-(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((expWW-tr-3746-4+sinnex-nn--nnexpWW-tr-7492-4+4-nsqrnexq+---------282)-384", "parentheses mismatch");
    test("fi.g", "don't know how to parse .g");
    test(
        "(nc7)sqrtE",
        "wlog a number/variable cannot be on the right",
    );
    test("", "empty string");
    test("++", "the last element cannot be an operator");
    test(
        "a12 (1)",
        "wlog a number/variable cannot be on the right of a closing paren",
    );
    test("++)", "operator cannot be on the left of a closing");
    test(")+12-(1+1) / (", "closing parentheses until position");
    test("12-()+(", "wlog an opening paren");
    test("12-() ())", "wlog an opening paren");
    test("12-(3-4)*2+ (1/2))", "closing parentheses until");
    test("12-(3-4)*2+ ((1/2)", "parentheses mismatch");
    test(r"5\6", r"don't know how to parse \");
    test(r"3.4.", r"don't know how to parse 3.4.");
    test(
        r"2sin({x})",
        r"number/variable cannot be on the left of a unary operator",
    );
}

#[test]
fn test_find_comma_op() {
    let pts = [
        ParsedToken::Paren(Paren::Close),
        ParsedToken::Paren(Paren::Close),
        ParsedToken::Op((
            0,
            Operator::make_bin(
                "atan2",
                crate::BinOp {
                    apply: |y: f64, x: f64| y.atan2(x),
                    prio: 1,
                    is_commutative: false,
                },
            ),
        )),
        ParsedToken::Paren(Paren::Open),
    ];
    assert_eq!(Some(2), find_op_of_comma(&pts));
    let pts = [
        ParsedToken::Paren(Paren::Close),
        ParsedToken::Op((
            0,
            Operator::make_bin(
                "atan2",
                crate::BinOp {
                    apply: |y: f64, x: f64| y.atan2(x),
                    prio: 1,
                    is_commutative: false,
                },
            ),
        )),
        ParsedToken::Paren(Paren::Open),
    ];
    assert_eq!(Some(1), find_op_of_comma(&pts));
}
