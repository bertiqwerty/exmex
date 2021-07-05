use num::Float;
use regex::{Regex, RegexSet};
use std::error::Error;
use std::fmt;
use std::str::FromStr;

use crate::types::BinOp;
use crate::types::{Expression, Node};

type VecOps<'a, T> = Vec<(&'a str, OperatorToken<T>)>;

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

fn make_operators<'a, T: Float>() -> (VecOps<'a, T>, String) {
    (
        [
            ("*", OperatorToken { bin_op: Some(BinOp{op: |a, b| a * b, prio: 1}), unary_op: None }),
            ("/", OperatorToken { bin_op: Some(BinOp{op: |a, b| a / b, prio: 1}), unary_op: None }),
            ("+", OperatorToken { bin_op: Some(BinOp{op: |a, b| a + b, prio: 0}), unary_op: Some(|a: T| a) }),
            ("-", OperatorToken { bin_op: Some(BinOp{op: |a, b| a - b, prio: 0}), unary_op: Some(|a: T| (-a)) }),
            ("sin", OperatorToken { bin_op: None, unary_op: Some(|a: T| a.sin()) }),
            ("cos", OperatorToken { bin_op: None, unary_op: Some(|a: T| a.cos()) }),
        ]
        .iter()
        .cloned()
        .collect(),
        r"cos|sin|[*/+\-]".to_string(),
    )
}

fn find_op<'a, T: Float>(name: &str, ops: &VecOps<'a, T>) -> OperatorToken<T> {
    ops.iter().find(|(op_name, _)| op_name == &name).unwrap().1
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum ParanToken {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct OperatorToken<T: Copy> {
    pub bin_op: Option<BinOp<T>>,
    pub unary_op: Option<fn(T) -> T>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum EvilToken<T: Float + FromStr> {
    Num(T),
    Paran(ParanToken),
    Op(OperatorToken<T>),
}

fn apply_regexes<T: Float + FromStr>(text: &str) -> Vec<EvilToken<T>>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let (ops, pattern_bin_ops) = make_operators::<T>();
    let pattern_nums = r"\.?[0-9]+(\.[0-9]+)?";
    let pattern_parans = r"\(|\)";
    let patterns = [pattern_nums, pattern_parans, pattern_bin_ops.as_str()];
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
                    ParanToken::Open
                } else if c == ')' {
                    ParanToken::Close
                } else {
                    panic!(
                        "Internal error. Paran {} is neither ( not ). Check the paran-regex.",
                        c
                    );
                })
            } else if matches.matched(2) {
                EvilToken::<T>::Op(find_op(elt_str, &ops))
            } else {
                panic!("Internal regex mismatch!");
            }
        })
        .collect()
}

fn make_expression<T>(tokens: &[EvilToken<T>], unary_op: Vec<fn(T) -> T>) -> Result<(Expression<T>, usize), EvilParseError>
where
    T: Float + FromStr + std::fmt::Debug
{
    fn unpack_binop<S>(bo: Option<BinOp<S>>) -> Result<BinOp<S>, EvilParseError>
    where
        S: Float + FromStr + std::fmt::Debug
    {
        match bo {
            Some(bo) => Ok(bo),
            None => Err(EvilParseError { msg: "Expected binary operator but there was None.".to_string() })
        }
    }
    let mut result = Expression::<T> {
        bin_ops: Vec::<BinOp<T>>::new(),
        nodes: Vec::<Node<T>>::new(),
        unary_ops: unary_op
    };
    let process_unary = |i: usize, uo| {
        match tokens[i+1] {
            EvilToken::Paran(p) => match p {
                ParanToken::Close => {
                    Err(
                        EvilParseError{
                            msg: "I do not understand a closing paran after an operator.".to_string()
                        }
                    )
                }
                ParanToken::Open => {
                    let (expr, i_forward) = make_expression::<T>(&tokens[i+2..], vec![uo])?;
                    Ok((Node::Expr(expr), i_forward + 2))
                }
            },
            EvilToken::Num(n) => {                                            
                Ok((Node::Num(uo(n)), 2))
            }
            EvilToken::Op(op) => {
                match op.unary_op {
                    Some(uo2) => {
                        match tokens[i+2] {
                            EvilToken::Paran(p) => match p {
                                ParanToken::Close => {
                                    Err(
                                        EvilParseError{
                                            msg: "I do not understand a closing paran after an operator.".to_string()
                                        }
                                    )
                                }
                                ParanToken::Open => {
                                    let (expr, i_forward) = make_expression::<T>(&tokens[i+3..], vec![uo, uo2])?;
                                    Ok((Node::Expr(expr), i_forward + 3))
                                }
                            }
                            _ => Err(
                                EvilParseError{
                                    msg: "If a unary operator follows a unary operator, e.g., -sin, we expect an ( as next token.".to_string()
                                }
                            )
                        }
                    },
                    None => Err(EvilParseError{msg: "A unary operator cannot be followed by a binary operator.".to_string()})
                }
            }
        }
    };

    let mut i: usize = 0;
    while i < tokens.len() {
        match tokens[i] {
            EvilToken::Op(b) => {
                match b.unary_op {
                    None => {
                        result.bin_ops.push(unpack_binop(b.bin_op)?);
                        i += 1;
                    },
                    Some(uo) => {
                        if i == 0 {
                            let (node, i_forward) = process_unary(i, uo)?;
                            result.nodes.push(node);
                            i += i_forward;
                        } else {
                            match tokens[i-1] {
                                EvilToken::Num(_) => {
                                    result.bin_ops.push(unpack_binop(b.bin_op)?);
                                    i += 1;
                                },
                                EvilToken::Paran(p) => match p {
                                    ParanToken::Open => {
                                        let msg = "Opening paran next to operator must not occur here.".to_string();
                                        return Err(EvilParseError{msg: msg});
                                    },
                                    ParanToken::Close => {
                                        result.bin_ops.push(unpack_binop(b.bin_op)?);
                                        i += 1;
                                    },
                                },
                                EvilToken::Op(_) => {
                                    let (node, i_forward) = process_unary(i, uo)?;
                                    result.nodes.push(node);
                                    i += i_forward;
                                }
                            }
                        }
                    }                    
                }
            }
            EvilToken::Num(n) => {
                result.nodes.push(Node::Num(n));
                i += 1;
            }
            EvilToken::Paran(p) => match p {
                ParanToken::Open => {
                    i += 1;
                    let (expr, i_forward) = make_expression::<T>(&tokens[i..], vec![])?;
                    result.nodes.push(Node::Expr(expr));
                    i += i_forward;
                }
                ParanToken::Close => {
                    i += 1;
                    break;
                }
            },
        }
    }

    Ok((result, i))
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
    let num_pred_succ = |idx: usize, forbidden: ParanToken| match expr_elts[idx] {
        EvilToken::Num(_) => Err(EvilParseError {
            msg: "A number cannot be next to a number.".to_string(),
        }),
        EvilToken::Paran(p) => {
            if p == forbidden {
                Err(EvilParseError {
                    msg: "A number cannot be on the right of a closing paran or on the left of an opening paran.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let binop_pred_succ = |idx: usize| match expr_elts[idx] {
        EvilToken::Op(op) =>{ if op.unary_op == None { Err(EvilParseError {
            msg: "A binary operator cannot be next to a binary operator.".to_string(),
        })}else { Ok(0) }},
        _ => Ok(0),
    };
    let paran_pred_succ = |idx: usize, forbidden: ParanToken| match expr_elts[idx] {
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
                        num_pred_succ(i + 1, ParanToken::Open)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, ParanToken::Close)?;
                    }
                    Ok(0)
                }
                EvilToken::Paran(p) => {
                    if i < expr_elts.len() - 1 {
                        match p {
                            ParanToken::Open => paran_pred_succ(i + 1, ParanToken::Close)?,
                            ParanToken::Close => paran_pred_succ(i + 1, ParanToken::Open)?,
                        };
                    }
                    open_paran_cnt += match p {
                        ParanToken::Close => -1,
                        ParanToken::Open => 1,
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
                            msg: "The last element cannot be an operator.".to_string()
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
    use crate::{parse::{parse}};

    #[test]
    fn test_preconditions() {
        fn test(sut: &str, msg_part: &str) {
            let err = parse::<f32>(sut);
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
        test("12 (", "number cannot be on the right of a closing");
        test("++)", "closing parantheses until");
        test(")12-(1+1) / (", "closing parantheses until position");
        test("12-()+(", "Wlog an opening paran");
        test("12-() ())", "Wlog an opening paran");
        test("12-(3-4)*2+ (1/2))", "closing parantheses until");
        test("12-(3-4)*2+ ((1/2)", "Parantheses mismatch.");
        test("--1", "e.g., -sin, we expect an (");
    }

    
}
