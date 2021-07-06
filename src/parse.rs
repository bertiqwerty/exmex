use num::Float;
use regex::{Regex, RegexSet};
use std::error::Error;
use std::fmt;
use std::iter::once;
use std::str::FromStr;

use crate::expression::BinOp;
use crate::expression::{Expression, Node};
use crate::util::apply_unary_ops;

type VecOps<'a, T> = Vec<(&'a str, OperatorPair<T>)>;

#[derive(Debug)]
pub struct EvilParseError {
    pub msg: String,
}
impl fmt::Display for EvilParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for EvilParseError {}

fn make_default_operators<'a, T: Float>() -> VecOps<'a, T> {
    [
        ("*", OperatorPair { bin_op: Some(BinOp{op: |a, b| a * b, prio: 1}), unary_op: None }),
        ("/", OperatorPair { bin_op: Some(BinOp{op: |a, b| a / b, prio: 1}), unary_op: None }),
        ("+", OperatorPair { bin_op: Some(BinOp{op: |a, b| a + b, prio: 0}), unary_op: Some(|a: T| a) }),
        ("-", OperatorPair { bin_op: Some(BinOp{op: |a, b| a - b, prio: 0}), unary_op: Some(|a: T| (-a)) }),
        ("sin", OperatorPair { bin_op: None, unary_op: Some(|a: T| a.sin()) }),
        ("cos", OperatorPair { bin_op: None, unary_op: Some(|a: T| a.cos()) }),
    ]
    .iter()
    .cloned()
    .collect()
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Paran {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct OperatorPair<T: Copy> {
    pub bin_op: Option<BinOp<T>>,
    pub unary_op: Option<fn(T) -> T>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum EvilToken<T: Float + FromStr> {
    Num(T),
    Paran(Paran),
    Op(OperatorPair<T>),
}

fn apply_regexes<T: Float + FromStr>(text: &str) -> Vec<EvilToken<T>>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let regex_escapes = r"|?^*+.\";
    let ops = make_default_operators::<T>();
    let pattern_ops = ops.iter().map(|(s, _)| {
        if regex_escapes.contains(s) {
            format!("\\{}", s)
        } else {
            s.to_string()
        }
    }).collect::<Vec<_>>().join("|");
    let pattern_nums = r"\.?[0-9]+(\.[0-9]+)?";
    let pattern_parans = r"\(|\)";
    let patterns = [pattern_nums, pattern_parans, pattern_ops.as_str()];
    let pattern_any = patterns.join("|");
    let any = Regex::new(pattern_any.as_str()).unwrap();

    let which_one = RegexSet::new(&patterns).unwrap();
    any.captures_iter(text)
        .map(|c| c[0].to_string())
        .map(|elt_string| {
            let elt_str = elt_string.as_str();
            let matches = which_one.matches(elt_str);
            if matches.matched(0) {
                EvilToken::<T>::Num(elt_str.parse::<T>().unwrap())
            } else if matches.matched(1) {
                let c = elt_str.chars().next().unwrap();
                EvilToken::<T>::Paran(if c == '(' {
                    Paran::Open
                } else if c == ')' {
                    Paran::Close
                } else {
                    panic!(
                        "Fatal. Paran {} is neither ( nor ). Check the paran-regex.",
                        c
                    );
                })
            } else if matches.matched(2) {
                let wrapped_op_token = ops.iter().find(|(op_name, _)| op_name == &elt_str);
                EvilToken::<T>::Op(match wrapped_op_token {
                    Some((_, op_token)) => *op_token,
                    None => {
                        panic!("Fatal. Could not find operator {}.", elt_str);
                    }
                })
            } else {
                panic!("Fatal. Internal regex mismatch!");
            }
        })
        .collect()
}

fn make_expression<T>(
    tokens: &[EvilToken<T>],
    unary_op: Vec<fn(T) -> T>,
) -> Result<(Expression<T>, usize), EvilParseError>
where
    T: Float + FromStr + std::fmt::Debug,
{
    fn unpack_binop<S>(bo: Option<BinOp<S>>) -> Result<BinOp<S>, EvilParseError>
    where
        S: Float + FromStr + std::fmt::Debug,
    {
        match bo {
            Some(bo) => Ok(bo),
            None => Err(EvilParseError {
                msg: "Expected binary operator but there was None.".to_string(),
            }),
        }
    }

    // this closure handles the case that a token is a unary operator and accesses the
    // variable tokens from the outer scope
    let process_unary = |i: usize, uo| {
        // gather subsequent unary operators from the beginning
        let uops = once(uo)
            .chain(
                (i + 1..tokens.len())
                    .map(|j| match tokens[j] {
                        EvilToken::Op(op) => op.unary_op,
                        _ => None,
                    })
                    .take_while(|uo_| uo_.is_some())
                    .flatten(),
            )
            .collect::<Vec<_>>();
        let n_uops = uops.len();

        match tokens[i + n_uops] {
            EvilToken::Paran(p) => match p {
                Paran::Close => Err(EvilParseError {
                    msg: "I do not understand a closing paran after an operator.".to_string(),
                }),
                Paran::Open => {
                    let (expr, i_forward) = make_expression::<T>(&tokens[i + n_uops + 1..], uops)?;
                    Ok((Node::Expr(expr), i_forward + n_uops + 1))
                }
            },
            EvilToken::Num(n) => Ok((Node::Num(apply_unary_ops(&uops, n)), n_uops + 1)),
            EvilToken::Op(_) => Err(EvilParseError {
                msg: "A unary operator cannot be followed by a binary operator.".to_string(),
            }),
        }
    };

    let mut result = Expression::<T> {
        bin_ops: Vec::<BinOp<T>>::new(),
        nodes: Vec::<Node<T>>::new(),
        unary_ops: unary_op,
    };

    // The main loop checks one token after the next whereby sub-expressions are
    // handled recursively. Thereby, the token-position-index idx_tkn is increased
    // according to the length of the sub-expression.
    let mut idx_tkn: usize = 0;
    while idx_tkn < tokens.len() {
        match tokens[idx_tkn] {
            EvilToken::Op(b) => match b.unary_op {
                None => {
                    result.bin_ops.push(unpack_binop(b.bin_op)?);
                    idx_tkn += 1;
                }
                Some(uo) => {
                    if idx_tkn == 0 {
                        let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                        result.nodes.push(node);
                        idx_tkn += idx_forward;
                    } else {
                        match tokens[idx_tkn - 1] {
                            EvilToken::Num(_) => {
                                result.bin_ops.push(unpack_binop(b.bin_op)?);
                                idx_tkn += 1;
                            }
                            EvilToken::Paran(p) => match p {
                                Paran::Open => {
                                    let msg = "Opening paran next to operator must not occur here."
                                        .to_string();
                                    return Err(EvilParseError { msg: msg });
                                }
                                Paran::Close => {
                                    result.bin_ops.push(unpack_binop(b.bin_op)?);
                                    idx_tkn += 1;
                                }
                            },
                            EvilToken::Op(_) => {
                                let (node, idx_forward) = process_unary(idx_tkn, uo)?;
                                result.nodes.push(node);
                                idx_tkn += idx_forward;
                            }
                        }
                    }
                }
            },
            EvilToken::Num(n) => {
                result.nodes.push(Node::Num(n));
                idx_tkn += 1;
            }
            EvilToken::Paran(p) => match p {
                Paran::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) = make_expression::<T>(&tokens[idx_tkn..], vec![])?;
                    result.nodes.push(Node::Expr(expr));
                    idx_tkn += i_forward;
                }
                Paran::Close => {
                    idx_tkn += 1;
                    break;
                }
            },
        }
    }
    Ok((result, idx_tkn))
}

fn check_preconditions<T>(expr_elts: &[EvilToken<T>]) -> Result<u8, EvilParseError>
where
    T: Float + FromStr + std::fmt::Debug,
{
    if expr_elts.len() == 0 {
        return Err(EvilParseError {
            msg: "Cannot parse empty string.".to_string(),
        });
    };
    let num_pred_succ = |idx: usize, forbidden: Paran| match expr_elts[idx] {
        EvilToken::Num(_) => Err(EvilParseError {
            msg: "A number cannot be next to a number.".to_string(),
        }),
        EvilToken::Paran(p) => {
            if p == forbidden {
                Err(EvilParseError {
                    msg: "Wlog, a number cannot be on the right of a closing paran.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let binop_pred_succ = |idx: usize| match expr_elts[idx] {
        EvilToken::Op(op) => {
            if op.unary_op == None {
                Err(EvilParseError {
                    msg: "A binary operator cannot be next to a binary operator.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let paran_pred_succ = |idx: usize, forbidden: Paran| match expr_elts[idx] {
        EvilToken::Paran(p) => {
            if p == forbidden {
                Err(EvilParseError {
                    msg: "Wlog an opening paran cannot be next to a closing paran.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let mut open_paran_cnt = 0i8;
    expr_elts
        .iter()
        .enumerate()
        .map(|(i, expr_elt)| -> Result<usize, EvilParseError> {
            match expr_elt {
                EvilToken::Num(_) => {
                    if i < expr_elts.len() - 1 {
                        num_pred_succ(i + 1, Paran::Open)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, Paran::Close)?;
                    }
                    Ok(0)
                }
                EvilToken::Paran(p) => {
                    if i < expr_elts.len() - 1 {
                        match p {
                            Paran::Open => paran_pred_succ(i + 1, Paran::Close)?,
                            Paran::Close => paran_pred_succ(i + 1, Paran::Open)?,
                        };
                    }
                    open_paran_cnt += match p {
                        Paran::Close => -1,
                        Paran::Open => 1,
                    };
                    if open_paran_cnt < 0 {
                        return Err(EvilParseError {
                            msg: format!("To many closing parantheses until position {}.", i)
                                .to_string(),
                        });
                    }
                    Ok(0)
                }
                EvilToken::Op(_) => {
                    if i < expr_elts.len() - 1 {
                        binop_pred_succ(i + 1)?;
                        Ok(0)
                    } else {
                        Err(EvilParseError {
                            msg: "The last element cannot be an operator.".to_string(),
                        })
                    }
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if open_paran_cnt != 0 {
        Err(EvilParseError {
            msg: "Parantheses mismatch.".to_string(),
        })
    } else {
        Ok(0)
    }
}

pub fn parse<T>(text: &str) -> Result<Expression<T>, EvilParseError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: Float + FromStr + std::fmt::Debug,
{
    let elts = apply_regexes::<T>(text);
    check_preconditions(&elts[..])?;
    let (expr, _) = make_expression(&elts[0..], vec![])?;
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use crate::parse::{apply_regexes, check_preconditions};

    #[test]
    fn test_preconditions() {
        fn test(text: &str, msg_part: &str) {
            let elts = apply_regexes::<f32>(text);
            let err = check_preconditions(&elts[..]);
            match err {
                Ok(_) => assert!(false),
                Err(e) => {
                    println!("{}", e.msg);
                    assert!(e.msg.contains(msg_part));
                }
            }
        }

        test("", "empty string.");
        test("++", "The last element cannot be an operator.");
        test(
            "12 (",
            "Wlog, a number cannot be on the right of a closing paran",
        );
        test("++)", "closing parantheses until");
        test(")12-(1+1) / (", "closing parantheses until position");
        test("12-()+(", "Wlog an opening paran");
        test("12-() ())", "Wlog an opening paran");
        test("12-(3-4)*2+ (1/2))", "closing parantheses until");
        test("12-(3-4)*2+ ((1/2)", "Parantheses mismatch.");
    }
}
