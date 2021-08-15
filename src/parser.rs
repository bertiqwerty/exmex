use crate::definitions::N_NODES_ON_STACK;
use crate::operators::Operator;
use lazy_static::lazy_static;
use regex::Regex;
use smallvec::SmallVec;
use std::error::Error;
use std::fmt::{self, Debug};
use std::str::FromStr;

/// This will be thrown at you if the parsing went wrong. Ok, obviously it is not an
/// exception, so thrown needs to be understood figuratively.
#[derive(Debug, Clone)]
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
pub fn tokenize_and_analyze<'a, T: Copy + FromStr + Debug, F: Fn(&'a str) -> Option<&'a str>>(
    text: &'a str,
    ops_in: &[Operator<'a, T>],
    is_numeric: F,
) -> Result<Vec<ParsedToken<'a, T>>, ExParseError>
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

    let mut res = Vec::new();
    res.reserve(2 * N_NODES_ON_STACK);

    for (i, c) in text.chars().enumerate() {
        if c == ' ' {
            cur_offset += 1;
        }
        if i == cur_offset && cur_offset < text.len() && c != ' ' {
            let maybe_op;
            let maybe_num;
            let maybe_name;
            let text_rest = &text[cur_offset..];
            let next_parsed_token = if c == '(' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Open)
            } else if c == ')' {
                cur_offset += 1;
                ParsedToken::<T>::Paren(Paren::Close)
            } else if c == '{' {
                let n_count = text_rest.chars().take_while(|c| *c != '}').count();
                cur_offset += n_count + 1;
                ParsedToken::<T>::Var(&text_rest[1..n_count])
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
                ParsedToken::<T>::Var(maybe_name.unwrap().as_str())
            } else {
                let msg = format!("how to parse the beginning of {}", text_rest);
                return Err(ExParseError { msg: msg });
            };
            res.push(next_parsed_token);
        }
    }
    check_preconditions(&res)?;
    Ok(res)
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
pub fn check_preconditions<T>(parsed_tokens: &[ParsedToken<T>]) -> Result<u8, ExParseError>
where
    T: Copy + FromStr + std::fmt::Debug,
{
    if parsed_tokens.len() == 0 {
        return Err(ExParseError {
            msg: "cannot parse empty string".to_string(),
        });
    };

    enum NeighborType {
        Predecessor,
        Successor,
    }

    let num_pred_succ = |idx: usize, neighbor_type: NeighborType| match &parsed_tokens[idx] {
        ParsedToken::Num(_) => Err(ExParseError {
            msg: "a number/variable cannot be next to a number/variable".to_string(),
        }),
        ParsedToken::Paren(p) => {
            let forbidden = match neighbor_type {
                NeighborType::Predecessor => Paren::Close,
                NeighborType::Successor => Paren::Open,
            };
            if p == &forbidden {
                Err(ExParseError {
                    msg: "wlog a number/variable cannot be on the right of a closing parenthesis"
                        .to_string(),
                })
            } else {
                Ok(0)
            }
        }
        ParsedToken::Op(op) => match neighbor_type {
            NeighborType::Predecessor => Ok(0),
            NeighborType::Successor => {
                if let Some(_) = op.bin_op {
                    Ok(0)
                } else if let Some(_) = op.unary_op {
                    Err(ExParseError {
                        msg: "a number/variable cannot be on the left of a unary operator"
                            .to_string(),
                    })
                } else {
                    Ok(0)
                }
            }
        },
        _ => Ok(0),
    };
    let op_pred_succ =
        |central_op: &Operator<T>, idx: usize, neighbor_type: NeighborType| match &parsed_tokens
            [idx]
        {
            ParsedToken::Op(neighbor_op) => {
                if neighbor_op.unary_op.is_none() && central_op.unary_op.is_none() {
                    Err(ExParseError {
                        msg: "a binary operator cannot be next to a binary operator".to_string(),
                    })
                } else if neighbor_op.unary_op.is_some() && central_op.unary_op.is_none() {
                    match neighbor_type {
                        NeighborType::Predecessor => Err(ExParseError {
                            msg: "a binary operator cannot be on the right of a unary"
                                .to_string(),
                        }),
                        _ => Ok(0),
                    }
                } else {
                    Ok(0)
                }
            }
            ParsedToken::Paren(paren) => match paren {
                Paren::Close => match neighbor_type {
                    NeighborType::Successor => Err(ExParseError {
                        msg: "an operator cannot be on the left of a closing paren".to_string(),
                    }),
                    NeighborType::Predecessor => {
                        if central_op.bin_op.is_some() {
                            Ok(0)
                        } else {
                            Err(ExParseError {
                                msg: "a unary operator cannot be on the right of a closing paren"
                                    .to_string(),
                            })
                        }
                    }
                },
                Paren::Open => {
                    if central_op.unary_op.is_none() {
                        match neighbor_type {
                            NeighborType::Predecessor => Err(ExParseError {
                                msg: "a binary operator cannot be on the right of an opening paren"
                                    .to_string(),
                            }),
                            NeighborType::Successor => Ok(0),
                        }
                    } else {
                        Ok(0)
                    }
                }
            },
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
    let mut open_paren_cnt = 0i32;
    parsed_tokens
        .iter()
        .enumerate()
        .map(|(i, expr_elt)| -> Result<usize, ExParseError> {
            match expr_elt {
                ParsedToken::Num(_) | ParsedToken::Var(_) => {
                    if i < parsed_tokens.len() - 1 {
                        num_pred_succ(i + 1, NeighborType::Successor)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, NeighborType::Predecessor)?;
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
                ParsedToken::Op(op) => {
                    if i < parsed_tokens.len() - 1 {
                        if i > 0 {
                            op_pred_succ(op, i - 1, NeighborType::Predecessor)?;
                        };
                        op_pred_succ(op, i + 1, NeighborType::Successor)?;
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
#[cfg(test)]
use crate::operators;
#[test]
fn test_apply_regexes() {
    let text = r"5\6";
    let ops = operators::make_default_operators::<f32>();
    let elts = tokenize_and_analyze(text, &ops, is_numeric_text);
    assert!(elts.is_err());
}

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
        fn check_err_msg<V>(err: Result<V, ExParseError>, msg_part: &str) {
            match err {
                Ok(_) => assert!(false),
                Err(e) => {
                    println!("{}", e.msg);
                    assert!(e.msg.contains(msg_part));
                }
            }
        }
        let ops = operators::make_default_operators::<f32>();
        let elts = tokenize_and_analyze(text, &ops, is_numeric_text);
        match elts {
            Ok(elts_unwr) => {
                let err = check_preconditions(&elts_unwr[..]);
                check_err_msg(err, msg_part);
            }
            Err(_) => check_err_msg(elts, msg_part),
        }
    }
    test("xo-17-(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((expWW-tr-3746-4+sinnex-nn--nnexpWW-tr-7492-4+4-nsqrnexq+---------282)-384", "parentheses mismatch");
    test("fi.g", "parse the beginning of .g");
    test("(nc7)sqrtE", "unary operator cannot be on the right");
    test("", "empty string");
    test("++", "the last element cannot be an operator");
    test(
        "a12 (",
        "wlog a number/variable cannot be on the right of a closing paren",
    );
    test("++)", "operator cannot be on the left of a closing");
    test(")12-(1+1) / (", "closing parentheses until position");
    test("12-()+(", "wlog an opening paren");
    test("12-() ())", "wlog an opening paren");
    test("12-(3-4)*2+ (1/2))", "closing parentheses until");
    test("12-(3-4)*2+ ((1/2)", "parentheses mismatch");
    test(r"5\6", r"how to parse the beginning of \");
    test(r"3 * log2 * 5", r"a binary operator cannot be on the right of a unary");
    test(r"3.4.", r"how to parse the beginning of 3.4.");
    test(
        r"3. .4",
        r"a number/variable cannot be next to a number/variable",
    );
    test(
        r"2sin({x})",
        r"number/variable cannot be on the left of a unary",
    );
}
