use crate::expression::{BinOpVec, DeepEx, DeepNode, FlatEx, N_NODES_ON_STACK};
use crate::operators::{make_default_operators, BinOp, Operator, UnaryOp, VecOfUnaryFuncs};
use itertools::Itertools;
use lazy_static::lazy_static;
use num::Float;
use regex::Regex;
use smallvec::SmallVec;
use std::error::Error;
use std::fmt::{self, Debug};
use std::iter::once;
use std::str::FromStr;

/// This will be thrown at you if the parsing went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Debug)]
pub struct ExParseError {
    pub msg: String,
}
impl fmt::Display for ExParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExParseError {}

#[derive(Debug, PartialEq, Eq)]
enum Paren {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq)]
enum ParsedToken<'a, T: Copy + FromStr> {
    Num(T),
    Paren(Paren),
    Op(Operator<'a, T>),
    Var(String),
}

fn is_numeric_text<'a>(text: &'a str) -> Option<&'a str> {
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
    if n_num_chars > 0 && n_dots < 2 {
        Some(&text[0..n_num_chars])
    } else {
        None
    }
}

fn is_numeric_regex<'a>(re: &Regex, text: &'a str) -> Option<&'a str> {
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
fn parsed_tokens<'a, 'b, T: Copy + FromStr + Debug, F: Fn(&'b str) -> Option<&'b str>>(
    text: &'b str,
    ops_in: &[Operator<'a, T>],
    is_numeric: F,
) -> Result<SmallVec<[ParsedToken<'a, T>; 2 * N_NODES_ON_STACK]>, ExParseError>
where
    <T as std::str::FromStr>::Err: Debug,
{
    // We sort operators inverse alphabetically such that log2 has higher priority than log (wlog :D).

    let mut ops_tmp = ops_in.iter().clone().collect::<SmallVec<[_; 64]>>();
    ops_tmp.sort_by(|o1, o2| o2.repr.partial_cmp(o1.repr).unwrap());
    let ops = ops_tmp; // from now on const

    lazy_static! {
        static ref RE_NAME: Regex = Regex::new(r"^[a-zA-Z_]+[a-zA-Z_0-9]*").unwrap();
    }

    let mut cur_offset = 0usize;
    let find_ops = |offset: usize| {
        ops.iter().find(|op| {
            let range_end = offset + op.repr.chars().count();
            if range_end > text.len() {
                false
            } else {
                op.repr == &text[offset..range_end]
            }
        })
    };

    let mut res = SmallVec::<[_; 2 * N_NODES_ON_STACK]>::new();

    for (i, c) in text.chars().enumerate() {
        if c == ' ' {
            cur_offset += 1;
        }
        if i == cur_offset && cur_offset < text.len() && c != ' ' {
            let maybe_op;
            let maybe_num;
            let maybe_name;
            let text_rest: &str = &text[cur_offset..];
            let next_parsed_token = if c == '(' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Open)
            } else if c == ')' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Close)
            } else if c == '{' {
                let n_count = text_rest.chars().take_while(|c| *c != '}').count();
                cur_offset += n_count + 1;
                ParsedToken::<T>::Var(text_rest[1..n_count + 1].to_string())
            } else if {
                maybe_num = is_numeric(text_rest);
                maybe_num.is_some()
            } {
                let num_str = maybe_num.unwrap();
                let n_chars = num_str.chars().count();
                cur_offset += n_chars;
                ParsedToken::<T>::Num(num_str.parse::<T>().unwrap())
            } else if {
                maybe_op = find_ops(cur_offset);
                maybe_op.is_some()
            } {
                let op = **maybe_op.unwrap();
                let n_chars = op.repr.chars().count();
                cur_offset += n_chars;
                ParsedToken::<T>::Op(op)
            } else if {
                maybe_name = RE_NAME.find(text_rest);
                maybe_name.is_some()
            } {
                let var_str = maybe_name.unwrap().as_str();
                let n_chars = var_str.chars().count();
                cur_offset += n_chars;
                ParsedToken::<T>::Var(maybe_name.unwrap().as_str().to_string())
            } else {
                let msg = format!("how to parse the beginning of {}", text_rest);
                return Err(ExParseError { msg: msg });
            };
            res.push(next_parsed_token);
        }
    }

    Ok(res)
}

/// Returns an expression that is created recursively and can be evaluated
///
/// # Arguments
///
/// * `parsed_tokens` - parsed tokens created with [`apply_regexes`]
/// * `parsed_vars` - elements of `parsed_tokens` that are variables
/// * `unary_ops` - unary operators of the expression to be build
///
/// # Errors
///
/// See [`parse_with_number_pattern`](parse_with_number_pattern)
///
fn make_expression<T>(
    parsed_tokens: &[ParsedToken<T>],
    parsed_vars: &[String],
    unary_ops: UnaryOp<T>,
) -> Result<(DeepEx<T>, usize), ExParseError>
where
    T: Copy + FromStr + Debug,
{
    fn unpack_binop<S>(bo: Option<BinOp<S>>) -> BinOp<S>
    where
        S: Copy + FromStr + Debug,
    {
        match bo {
            Some(bo) => bo,
            None => panic!("This is probably a bug. Expected binary operator but there was none."),
        }
    }

    let find_var_index = |name: &str| {
        let idx = parsed_vars
            .iter()
            .enumerate()
            .find(|(_, n)| n.as_str() == name);
        match idx {
            Some((i, _)) => i,
            None => {
                panic!("This is probably a bug. I don't know variable {}", name)
            }
        }
    };
    // this closure handles the case that a token is a unary operator and accesses the
    // variable 'tokens' from the outer scope
    let process_unary = |i: usize, uo| {
        // gather subsequent unary operators from the beginning
        let vec_of_uops = once(uo)
            .chain(
                (i + 1..parsed_tokens.len())
                    .map(|j| match parsed_tokens[j] {
                        ParsedToken::Op(op) => op.unary_op,
                        _ => None,
                    })
                    .take_while(|uo_| uo_.is_some())
                    .flatten(),
            )
            .collect::<VecOfUnaryFuncs<_>>();
        let n_uops = vec_of_uops.len();
        let uop = UnaryOp::from_vec(vec_of_uops);
        match &parsed_tokens[i + n_uops] {
            ParsedToken::Paren(p) => match p {
                Paren::Close => Err(ExParseError {
                    msg: "closing parenthesis after an operator".to_string(),
                }),
                Paren::Open => {
                    let (expr, i_forward) =
                        make_expression::<T>(&parsed_tokens[i + n_uops + 1..], &parsed_vars, uop)?;
                    Ok((DeepNode::Expr(expr), i_forward + n_uops + 1))
                }
            },
            ParsedToken::Var(name) => {
                let expr = DeepEx::new(
                    vec![DeepNode::Var(find_var_index(&name))],
                    BinOpVec::new(),
                    uop,
                )?;
                Ok((DeepNode::Expr(expr), n_uops + 1))
            }
            ParsedToken::Num(n) => Ok((DeepNode::Num(uop.apply(*n)), n_uops + 1)),
            ParsedToken::Op(_) => Err(ExParseError {
                msg: "a unary operator cannot be followed by a binary operator".to_string(),
            }),
        }
    };

    let mut bin_ops = BinOpVec::new();
    let mut nodes = Vec::<DeepNode<T>>::new();

    // The main loop checks one token after the next whereby sub-expressions are
    // handled recursively. Thereby, the token-position-index idx_tkn is increased
    // according to the length of the sub-expression.
    let mut idx_tkn: usize = 0;
    while idx_tkn < parsed_tokens.len() {
        match &parsed_tokens[idx_tkn] {
            ParsedToken::Op(b) => match b.unary_op {
                None => {
                    bin_ops.push(unpack_binop(b.bin_op));
                    idx_tkn += 1;
                }
                Some(uo) => {
                    // might the operator be unary?
                    if idx_tkn == 0 {
                        // if the first element is an operator it must be unary
                        let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                        nodes.push(node);
                        idx_tkn += idx_forward;
                    } else {
                        // decide type of operator based on predecessor
                        match &parsed_tokens[idx_tkn - 1] {
                            ParsedToken::Num(_) | ParsedToken::Var(_) => {
                                // number or variable as predecessor means binary operator
                                bin_ops.push(unpack_binop(b.bin_op));
                                idx_tkn += 1;
                            }
                            ParsedToken::Paren(p) => match p {
                                Paren::Open => {
                                    let msg = "This is probably a bug. An opening paren cannot be the predecessor of a binary operator.";
                                    panic!("{}", msg);
                                }
                                Paren::Close => {
                                    bin_ops.push(unpack_binop(b.bin_op));
                                    idx_tkn += 1;
                                }
                            },
                            ParsedToken::Op(_) => {
                                let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                                nodes.push(node);
                                idx_tkn += idx_forward;
                            }
                        }
                    }
                }
            },
            ParsedToken::Num(n) => {
                nodes.push(DeepNode::Num(*n));
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                nodes.push(DeepNode::Var(find_var_index(&name)));
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => match p {
                Paren::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) = make_expression::<T>(
                        &parsed_tokens[idx_tkn..],
                        &parsed_vars,
                        UnaryOp::new(),
                    )?;
                    nodes.push(DeepNode::Expr(expr));
                    idx_tkn += i_forward;
                }
                Paren::Close => {
                    idx_tkn += 1;
                    break;
                }
            },
        }
    }
    Ok((DeepEx::new(nodes, bin_ops, unary_ops)?, idx_tkn))
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
fn check_preconditions<T>(parsed_tokens: &[ParsedToken<T>]) -> Result<u8, ExParseError>
where
    T: Copy + FromStr + std::fmt::Debug,
{
    if parsed_tokens.len() == 0 {
        return Err(ExParseError {
            msg: "cannot parse empty string".to_string(),
        });
    };
    let num_pred_succ = |idx: usize, forbidden: Paren| match &parsed_tokens[idx] {
        ParsedToken::Num(_) => Err(ExParseError {
            msg: "a number/variable cannot be next to a number/variable".to_string(),
        }),
        ParsedToken::Paren(p) => {
            if p == &forbidden {
                Err(ExParseError {
                    msg: "wlog a number/variable cannot be on the right of a closing parenthesis"
                        .to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let binop_pred_succ = |idx: usize| match parsed_tokens[idx] {
        ParsedToken::Op(op) => {
            if op.unary_op == None {
                Err(ExParseError {
                    msg: "a binary operator cannot be next to a binary operator".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let paren_pred_succ = |idx: usize, forbidden: Paren| match &parsed_tokens[idx] {
        ParsedToken::Paren(p) => {
            if p == &forbidden {
                Err(ExParseError {
                    msg: "wlog an opening paren cannot be next to a closing paren".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let mut open_paren_cnt = 0i8;
    parsed_tokens
        .iter()
        .enumerate()
        .map(|(i, expr_elt)| -> Result<usize, ExParseError> {
            match expr_elt {
                ParsedToken::Num(_) | ParsedToken::Var(_) => {
                    if i < parsed_tokens.len() - 1 {
                        num_pred_succ(i + 1, Paren::Open)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, Paren::Close)?;
                    }
                    Ok(0)
                }
                ParsedToken::Paren(p) => {
                    if i < parsed_tokens.len() - 1 {
                        match p {
                            Paren::Open => paren_pred_succ(i + 1, Paren::Close)?,
                            Paren::Close => paren_pred_succ(i + 1, Paren::Open)?,
                        };
                    }
                    open_paren_cnt += match p {
                        Paren::Close => -1,
                        Paren::Open => 1,
                    };
                    if open_paren_cnt < 0 {
                        return Err(ExParseError {
                            msg: format!("too many closing parentheses until position {}", i)
                                .to_string(),
                        });
                    }
                    Ok(0)
                }
                ParsedToken::Op(_) => {
                    if i < parsed_tokens.len() - 1 {
                        binop_pred_succ(i + 1)?;
                        Ok(0)
                    } else {
                        Err(ExParseError {
                            msg: "the last element cannot be an operator".to_string(),
                        })
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if open_paren_cnt != 0 {
        Err(ExParseError {
            msg: "parentheses mismatch".to_string(),
        })
    } else {
        Ok(0)
    }
}

fn parsed_tokens_to_flatex<T: Copy + FromStr + Debug>(
    parsed_tokens: &SmallVec<[ParsedToken<T>; 2 * N_NODES_ON_STACK]>,
) -> Result<FlatEx<T>, ExParseError> {
    let parsed_vars = parsed_tokens
        .iter()
        .filter_map(|pt| match pt {
            ParsedToken::Var(name) => Some(name.clone()),
            _ => None,
        })
        .unique()
        .collect::<SmallVec<[_; N_NODES_ON_STACK]>>();

    check_preconditions(&parsed_tokens[..])?;

    let (expr, _) = make_expression(&parsed_tokens[0..], &parsed_vars, UnaryOp::new())?;
    Ok(expr.flatten())
}

/// Parses a string and a vector of operators into an expression that can be evaluated.
///
/// # Errors
///
/// An error is returned in case [`parse_with_number_pattern`](parse_with_number_pattern)
/// returns one.
pub fn parse<'a, T>(text: &str, ops: &[Operator<'a, T>]) -> Result<FlatEx<T>, ExParseError>
where
    <T as std::str::FromStr>::Err: Debug,
    T: Copy + FromStr + Debug,
{
    let parsed_tokens = parsed_tokens(text, ops, is_numeric_text)?;
    parsed_tokens_to_flatex(&parsed_tokens)
}

/// Parses a string and a vector of operators and a regex pattern that defines the looks
/// of a number into an expression that can be evaluated.
///
/// # Errors
///
/// An [`ExParseError`](ExParseError) is returned, if
///
//
// from apply_regexes
//
/// * the argument `number_regex_pattern` cannot be compiled,
/// * the argument `text` contained a character that did not match any regex (e.g.,
///   if there is a `Δ` in `text` but no [operator](Operator) with
///   [`repr`](Operator::repr) equal to `Δ` is given),
//
// from check_preconditions
//
/// * the to-be-parsed string is empty,
/// * a number or variable is next to another one, e.g., `2 {x}`,
/// * wlog a number or variable is on the right of a closing parenthesis, e.g., `)5`,
/// * a binary operator is next to another binary operator, e.g., `2*/4`,
/// * wlog a closing parenthesis is next to an opening one, e.g., `)(` or `()`,
/// * too many closing parentheses at some position, e.g., `(4+6) - 5)*2`,
/// * the last element is an operator, e.g., `1+`,
/// * the number of opening and closing parenthesis do not match, e.g., `((4-2)`,
//
// from make_expression
//
/// * in `parsed_tokens` a closing parentheses is directly following an operator, e.g., `+)`, or
/// * a unary operator is followed directly by a binary operator, e.g., `sin*`.
///
pub fn parse_with_number_pattern<'a, 'b, T>(
    text: &'b str,
    ops: &[Operator<'a, T>],
    number_regex_pattern: &str,
) -> Result<FlatEx<T>, ExParseError>
where
    <T as std::str::FromStr>::Err: Debug,
    T: Copy + FromStr + Debug,
{
    let beginning_number_regex_regex = format!("^({})", number_regex_pattern);
    let re_number = match Regex::new(beginning_number_regex_regex.as_str()) {
        Ok(regex) => regex,
        Err(_) => {
            return Err(ExParseError {
                msg: "Cannot compile the passed number regex.".to_string(),
            })
        }
    };
    let is_numeric = |text: &'b str| is_numeric_regex(&re_number, &text);
    let parsed_tokens = parsed_tokens(text, ops, is_numeric)?;
    parsed_tokens_to_flatex(&parsed_tokens)
}

/// Parses a string into an expression that can be evaluated using default operators.
///
/// # Errors
///
/// An error is returned in case [`parse`](parse)
/// returns one.
pub fn parse_with_default_ops<T>(text: &str) -> Result<FlatEx<T>, ExParseError>
where
    <T as std::str::FromStr>::Err: Debug,
    T: Float + FromStr + Debug,
{
    let ops = make_default_operators::<T>();
    Ok(parse(&text, &ops)?)
}

#[cfg(test)]
mod tests {
    use crate::{
        parse::{check_preconditions, is_numeric_text, make_default_operators, parsed_tokens},
        ExParseError,
    };

    #[test]
    fn test_apply_regexes() {
        let text = r"5\6";
        let ops = make_default_operators::<f32>();
        let elts = parsed_tokens(text, &ops, is_numeric_text);
        assert!(elts.is_err());
    }

    #[test]
    fn test_preconditions() {
        fn test(text: &str, msg_part: &str) {
            fn check_err_msg<V>(err: Result<V, ExParseError>, msg_part: &str) {
                match err {
                    Ok(_) => assert!(false),
                    Err(e) => {
                        println!("{}", e.msg);
                        assert!(e.msg.contains(msg_part));
                    }
                }
            }
            let ops = make_default_operators::<f32>();
            let elts = parsed_tokens(text, &ops, is_numeric_text);
            match elts {
                Ok(elts_unwr) => {
                    let err = check_preconditions(&elts_unwr[..]);
                    check_err_msg(err, msg_part);
                }
                Err(_) => check_err_msg(elts, msg_part),
            }
        }

        test("", "empty string");
        test("++", "the last element cannot be an operator");
        test(
            "a12 (",
            "wlog a number/variable cannot be on the right of a closing paren",
        );
        test("++)", "closing parentheses until");
        test(")12-(1+1) / (", "closing parentheses until position");
        test("12-()+(", "wlog an opening paren");
        test("12-() ())", "wlog an opening paren");
        test("12-(3-4)*2+ (1/2))", "closing parentheses until");
        test("12-(3-4)*2+ ((1/2)", "parentheses mismatch");
        test(r"5\6", r"how to parse the beginning of \");
        test(r"3 * log2 * 5", r"binary operator cannot be next");
        test(r"3.4.", r"how to parse the beginning of 3.4.");
        test(
            r"3. .4",
            r"a number/variable cannot be next to a number/variable",
        );
    }
}
