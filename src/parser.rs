use crate::definitions::N_NODES_ON_STACK;
use crate::{operators::Operator, ExError, ExResult};
use lazy_static::lazy_static;
use regex::Regex;
use smallvec::SmallVec;
use std::fmt::Debug;
use std::str::FromStr;

#[derive(Debug, PartialEq, Eq)]
pub enum Paren {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParsedToken<'a, T: Copy + FromStr> {
    Num(T),
    Paren(Paren),
    Op(Operator<'a, T>),
    Var(&'a str),
}

pub fn is_numeric_text<'a>(text: &'a str) -> Option<&'a str> {
    let mut n_dots = 0;
    let n_num_chars = text
        .chars()
        .take_while(|c| {
            let is_dot = *c == '.';
            if is_dot {
                n_dots += 1;
            }
            c.is_digit(10) || is_dot
        })
        .count();
    if (n_num_chars > 1 && n_dots < 2) || (n_num_chars == 1 && n_dots == 0) {
        Some(&text[0..n_num_chars])
    } else {
        None
    }
}

pub fn is_numeric_regex<'a>(re: &Regex, text: &'a str) -> Option<&'a str> {
    let maybe_num = re.find(text);
    match maybe_num {
        Some(m) => Some(m.as_str()),
        None => None,
    }
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
    T: Copy + FromStr + Debug,
    F: Fn(&'a str) -> Option<&'a str>,
{
    // Make sure that the text does not contain unicode characters
    if text.chars().any(|c| !c.is_ascii()) {
        return Err(ExError {
            msg: "only ascii characters are supported".to_string(),
        });
    };

    // We sort operators inverse alphabetically such that log2 has higher priority than log (wlog :D).
    let mut ops_tmp = ops_in.iter().clone().collect::<SmallVec<[_; 64]>>();
    ops_tmp.sort_unstable_by(|o1, o2| o2.repr().partial_cmp(o1.repr()).unwrap());
    let ops = ops_tmp; // from now on const

    lazy_static! {
        static ref RE_VAR_NAME: Regex = Regex::new(r"^[a-zA-Z_]+[a-zA-Z_0-9]*").unwrap();
    }
    lazy_static! {
        static ref RE_VAR_NAME_EXACT: Regex = Regex::new(r"^[a-zA-Z_]+[a-zA-Z_0-9]*$").unwrap();
    }

    let find_ops = |offset: usize| {
        ops.iter().find(|op| {
            let range_end = offset + op.repr().chars().count();
            if range_end > text.len() {
                false
            } else if op.repr() == &text[offset..range_end] {
                if  !op.has_bin() && range_end < text.len() && RE_VAR_NAME_EXACT.is_match(&text[offset..range_end + 1]) {
                    false
                } else {
                    true
                }
            } else {
                false
            }
        })
    };
    let mut res: SmallVec<[_; N_NODES_ON_STACK]> = SmallVec::new();
    let mut cur_offset = 0usize;
    for (i, c) in text.chars().enumerate() {
        if c == ' ' {
            cur_offset += 1;
        } else if i == cur_offset && cur_offset < text.len() {
            let text_rest = &text[cur_offset..];
            let next_parsed_token = if c == '(' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Open)
            } else if c == ')' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Close)
            } else if c == '{' {
                let n_count = text_rest.chars().take_while(|c| *c != '}').count();
                let var_name = &text_rest[1..n_count];
                let n_spaces = var_name.chars().filter(|c| *c == ' ').count();
                // we need to subtract spaces from the offset, since they are added in the first if again.
                cur_offset += n_count + 1 - n_spaces;
                ParsedToken::<T>::Var(var_name)
            } else if let Some(num_str) = is_numeric(text_rest) {
                let n_chars = num_str.chars().count();
                cur_offset += n_chars;
                ParsedToken::<T>::Num(num_str.parse::<T>().map_err(
                    |e| ExError{
                        msg: format!("could not parse '{}', {:?}", num_str, e)
                    })?
                )
            } else if let Some(op) = find_ops(cur_offset) {
                let n_chars = op.repr().chars().count();
                cur_offset += n_chars;
                match op.constant() {
                    Some(constant) => ParsedToken::<T>::Num(constant),
                    None => ParsedToken::<T>::Op(**op),
                }
            } else if let Some(var_str) = RE_VAR_NAME.find(text_rest) {
                let var_str = var_str.as_str();
                let n_chars = var_str.chars().count();
                cur_offset += n_chars;
                ParsedToken::<T>::Var(var_str)
            } else {
                let msg = format!("how to parse the beginning of {}", text_rest);
                return Err(ExError { msg });
            };
            res.push(next_parsed_token);
        }
    }
    Ok(res)
}

struct PairPreCondition<'a, T: Copy + FromStr> {
    apply: fn(&ParsedToken<'a, T>, &ParsedToken<'a, T>) -> ExResult<()>,
}

fn make_pair_pre_conditions<'a, 'b, T: Copy + FromStr + Debug>() -> [PairPreCondition<'a, T>; 9] {
    [
        PairPreCondition {
            apply: |left, right| {
                let num_var_str =
                    "a number/variable cannot be next to a number/variable, violated by ";
                match (left, right) {
                    (ParsedToken::Num(num), ParsedToken::Var(var))
                    | (ParsedToken::Var(var), ParsedToken::Num(num)) => Err(ExError {
                        msg: format!("{}'{:?}' and '{}'", num_var_str, num, var),
                    }),
                    (ParsedToken::Num(num1), ParsedToken::Num(num2)) => Err(ExError {
                        msg: format!("{}'{:?}' and '{:?}'", num_var_str, num1, num2),
                    }),
                    (ParsedToken::Var(var1), ParsedToken::Var(var2)) => Err(ExError {
                        msg: format!("{}'{:?}' and '{:?}'", num_var_str, var1, var2),
                    }),
                    _ => Ok(()),
                }
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Num(_))
                | (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Var(_))
                | (ParsedToken::Num(_), ParsedToken::Paren(_p @ Paren::Open))
                | (ParsedToken::Var(_), ParsedToken::Paren(_p @ Paren::Open)) => Err(ExError {
                    msg: "wlog a number/variable cannot be on the right of a closing parenthesis"
                        .to_string(),
                }),
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Num(_), ParsedToken::Op(op))
                | (ParsedToken::Var(_), ParsedToken::Op(op))
                    // we do not ask for is_unary since operators can be both
                    if !op.has_bin() =>
                {
                    Err(ExError{
                        msg: "a number/variable cannot be on the left of a unary operator".to_string()
                    })
                }
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                (ParsedToken::Op(op_l), ParsedToken::Op(op_r))
                    if !op_l.has_unary() && !op_r.has_unary() =>
                {
                    Err(ExError{
                        msg:format!(
                            "a binary operator cannot be next to the binary operator, violated by '{}' left of '{}'", 
                            op_l.repr(), 
                            op_r.repr()
                        )
                    })
                }
                _ => Ok(()),
            }
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                (ParsedToken::Op(op_l), ParsedToken::Op(op_r))
                    if !op_l.has_bin() && !op_r.has_unary() =>
                {
                    Err(ExError{
                        msg:format!(
                            "a unary operator cannot be on the left of a binary one, violated by '{}' left of '{}'", 
                            op_l.repr(), 
                            op_r.repr()
                        )
                    })
                }
                _ => Ok(()),
            }
            },
        },
        PairPreCondition {
            apply: |left, right| match (left, right) {
                (ParsedToken::Op(op), ParsedToken::Paren(_p @ Paren::Close)) => Err(ExError {
                    msg: format!(
                        "an operator cannot be on the left of a closing paren, violated by '{}'",
                        op.repr()
                    ),
                }),
                _ => Ok(()),
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                (ParsedToken::Paren(_p @ Paren::Close), ParsedToken::Op(op)) if !op.has_bin() => {
                    Err(ExError{
                        msg:format!(
                            "a unary operator cannot be on the right of a closing paren, violated by '{}'", 
                            op.repr()
                        )
                    })
                }
                _ => Ok(()),
            }
            },
        },
        PairPreCondition {
            apply: |left, right| {
                match (left, right) {
                (ParsedToken::Paren(_p @ Paren::Open), ParsedToken::Op(op)) if !op.has_unary() => {
                    Err(ExError{
                        msg:format!("a binary operator cannot be on the right of an opening paren, violated by '{}'", op.repr())
                    })
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
                ) => Err(ExError {
                    msg: "wlog an opening paren cannot be next to a closing paren".to_string(),
                }),
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
pub fn check_parsed_token_preconditions<'a, T>(parsed_tokens: &[ParsedToken<'a, T>]) -> ExResult<()>
where
    T: Copy + FromStr + Debug,
{
    if parsed_tokens.len() == 0 {
        return Err(ExError {
            msg: "cannot parse empty string".to_string(),
        });
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
                        return Err(ExError {
                            msg: format!("too many closing parentheses until position {}", i),
                        });
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        })
        .collect::<ExResult<Vec<_>>>()?;
    if open_paren_cnt != 0 {
        Err(ExError {
            msg: "parentheses mismatch".to_string(),
        })
    } else if let ParsedToken::Op(_) = parsed_tokens[parsed_tokens.len() - 1] {
        Err(ExError {
            msg: "the last element cannot be an operator".to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
use crate::operators::{DefaultOpsFactory, MakeOperators};

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

                    assert!(false)
                }
                Err(e) => {
                    println!("msg '{}' should contain '{}'", e.msg, msg_part);
                    assert!(e.msg.contains(msg_part));
                }
            }
        }
        let ops = DefaultOpsFactory::<f32>::make();
        let elts = tokenize_and_analyze(text, &ops, is_numeric_text);
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
    test("fi.g", "parse the beginning of .g");
    test("(nc7)sqrtE", "wlog a number/variable cannot be on the right");
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
    test(r"5\6", r"how to parse the beginning of \");
    test(r"3.4.", r"how to parse the beginning of 3.4.");
    test(
        r"3. .4",
        r"a number/variable cannot be next to a number/variable",
    );
    test(
        r"2sin({x})",
        r"number/variable cannot be on the left of a unary operator",
    );
}
