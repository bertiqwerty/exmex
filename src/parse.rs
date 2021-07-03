use num::Float;
use regex::{Regex, RegexSet};
use std::error::Error;
use std::fmt;
use std::str::FromStr;

use crate::types::{BinaryOperator, Expression, Node};

type VecBinOps<'a, T> = Vec<(&'a str, BinaryOperator<T>)>;

#[derive(Debug)]
pub struct ExprParseError {
    pub msg: String,
}
impl fmt::Display for ExprParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
impl Error for ExprParseError {}

fn make_binary_operators<'a, T: Float>() -> (VecBinOps<'a, T>, String) {
    (
        [
            ("*", BinaryOperator { f: |a, b| a * b, priority: 1 }),
            ("/", BinaryOperator { f: |a, b| a / b, priority: 1 }),
            ("+", BinaryOperator { f: |a, b| a + b, priority: 0 }),
            ("-", BinaryOperator { f: |a, b| a - b, priority: 0 }),
        ]
        .iter()
        .cloned()
        .collect(),
        r"[*/+\-]".to_string(),
    )
}

fn find_op<'a, T: Float>(name: &str, ops: &VecBinOps<'a, T>) -> BinaryOperator<T> {
    ops.iter().find(|(op_name, _)| op_name == &name).unwrap().1
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Paran {
    Open,
    Close,
}

#[derive(Debug, PartialEq, Eq, Clone)]
enum ExprElt<T: Float + FromStr> {
    Num(T),
    Paran(Paran),
    BinOp(BinaryOperator<T>),
}

fn apply_regexes<T: Float + FromStr>(text: &str) -> Vec<ExprElt<T>>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let (bin_ops, pattern_bin_ops) = make_binary_operators::<T>();
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
                ExprElt::<T>::Num(elt_str.parse::<T>().unwrap())
            } else if matches.matched(1) {
                let c = elt_str.chars().next().unwrap();
                ExprElt::<T>::Paran(if c == '(' {
                    Paran::Open
                } else if c == ')' {
                    Paran::Close
                } else {
                    panic!(
                        "Internal error. Paran {} is neither ( not ). Check the paran-regex.",
                        c
                    );
                })
            } else if matches.matched(2) {
                ExprElt::<T>::BinOp(find_op(elt_str, &bin_ops))
            } else {
                panic!("Internal regex mismatch!");
            }
        })
        .collect()
}

fn make_expression<T>(expr_elts: &[ExprElt<T>]) -> (Expression<T>, usize)
where
    T: Float + FromStr + std::fmt::Debug,
{
    let mut result = Expression::<T> {
        bin_ops: Vec::<BinaryOperator<T>>::new(),
        nodes: Vec::<Node<T>>::new(),
    };
    let mut i: usize = 0;
    while i < expr_elts.len() {
        match expr_elts[i] {
            ExprElt::BinOp(b) => {
                result.bin_ops.push(b);
                i += 1;
            }
            ExprElt::Num(n) => {
                result.nodes.push(Node::Num(n));
                i += 1;
            }
            ExprElt::Paran(p) => match p {
                Paran::Open => {
                    i += 1;
                    let (expr, i_forward) = make_expression::<T>(&expr_elts[i..]);
                    result.nodes.push(Node::Expr(expr));
                    i += i_forward;
                }
                Paran::Close => {
                    i += 1;
                    break;
                }
            },
        }
    }
    (result, i)
}

fn check_preconditions<T>(expr_elts: &[ExprElt<T>]) -> Result<u8, ExprParseError>
where
    T: Float + FromStr + std::fmt::Debug,
{
    if expr_elts.len() == 0 {
        return Err(ExprParseError {
            msg: "Cannot parse empty string.".to_string(),
        });
    };
    let num_pred_succ = |idx: usize, forbidden: Paran| match expr_elts[idx] {
        ExprElt::Num(_) => Err(ExprParseError {
            msg: "A number cannot be next to a number.".to_string(),
        }),
        ExprElt::Paran(p) => {
            if p == forbidden {
                Err(ExprParseError {
                    msg: "A number cannot be on the right of a closing paran or on the left of an opening paran.".to_string(),
                })
            } else {
                Ok(0)
            }
        }
        _ => Ok(0),
    };
    let binop_pred_succ = |idx: usize| match expr_elts[idx] {
        ExprElt::BinOp(_) => Err(ExprParseError {
            msg: "A binary operator cannot be next to a binary operator.".to_string(),
        }),
        _ => Ok(0),
    };
    let paran_pred_succ = |idx: usize, forbidden: Paran| match expr_elts[idx] {
        ExprElt::Paran(p) => {
            if p == forbidden {
                Err(ExprParseError {
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
        .map(|(i, expr_elt)| -> Result<usize, ExprParseError> {
            match expr_elt {
                ExprElt::Num(_) => {
                    if i < expr_elts.len() - 1 {
                        num_pred_succ(i + 1, Paran::Open)?;
                    }
                    if i > 0 {
                        num_pred_succ(i - 1, Paran::Close)?;
                    }
                    Ok(0)
                }
                ExprElt::Paran(p) => {
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
                        return Err(ExprParseError {
                            msg: format!("To many closing parantheses until position {}.", i)
                                .to_string(),
                        });
                    }
                    Ok(0)
                }
                ExprElt::BinOp(_) => {
                    if i < expr_elts.len() - 1 {
                        binop_pred_succ(i + 1)?;
                    }
                    Ok(0)
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if open_paran_cnt != 0 {
        Err(ExprParseError {
            msg: "Parantheses mismatch.".to_string(),
        })
    } else {
        Ok(0)
    }
}

pub fn parse<T>(text: &str) -> Result<Expression<T>, ExprParseError>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
    T: Float + FromStr + std::fmt::Debug,
{
    let elts = apply_regexes::<T>(text);
    check_preconditions(&elts[..])?;
    let (expr, _) = make_expression(&elts[0..]);
    Ok(expr)
}

#[cfg(test)]
mod tests {
    use itertools::izip;

    use crate::{
        parse::{
            apply_regexes, find_op, make_binary_operators, make_expression, parse, ExprElt,
            Expression, Node, Paran,
        },
        types::BinaryOperator,
        util::assert_float_eq,
    };

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
        test("++", "binary operator cannot be next to a binary");
        test("12 (", "number cannot be on the right of a closing");
        test(")12-(1+1) / (", "closing parantheses until position");
        test("12-()+(", "Wlog an opening paran");
        test("12-() ())", "Wlog an opening paran");
        test("12-(3-4)*2+ (1/2))", "closing parantheses until");
        test("12-(3-4)*2+ ((1/2)", "Parantheses mismatch.");
    }

    fn check_num(expr: &Expression<f32>, idx: usize, reference: f32) {
        match expr.nodes[idx] {
            Node::<f32>::Num(n) => assert_float_eq(n, reference),
            _ => assert!(false),
        }
    }
    fn check_bin_op(
        expr: &Expression<f32>,
        idx: usize,
        reference: usize,
        bin_ops: &Vec<(&str, BinaryOperator<f32>)>,
    ) {
        assert_eq!(expr.bin_ops[idx], bin_ops[reference].1);
    }

    #[test]
    fn test_make_flat_expression() {
        let (bin_ops, _) = make_binary_operators::<f32>();

        let expr_elts = vec![
            ExprElt::<f32>::Num(1.0),
            ExprElt::<f32>::BinOp(bin_ops[0].1),
            ExprElt::<f32>::Num(2.0),
        ];
        let (expr, i) = make_expression::<f32>(&expr_elts[0..]);
        assert_eq!(i, 3);
        check_num(&expr, 0, 1.0);
        check_num(&expr, 1, 2.0);
        check_bin_op(&expr, 0, 0, &bin_ops);
    }
    #[test]
    fn test_make_parans_first_expression() {
        let (bin_ops, _) = make_binary_operators::<f32>();

        let expr_elts = vec![
            ExprElt::<f32>::Num(1.0),
            ExprElt::<f32>::BinOp(bin_ops[0].1),
            ExprElt::<f32>::Num(2.0),
            ExprElt::<f32>::BinOp(bin_ops[1].1),
            ExprElt::<f32>::Paran(Paran::Open),
            ExprElt::<f32>::Num(3.0),
            ExprElt::<f32>::BinOp(bin_ops[2].1),
            ExprElt::<f32>::Num(4.0),
            ExprElt::<f32>::Paran(Paran::Close),
        ];
        let (expr, _) = make_expression::<f32>(&expr_elts[0..]);
        match &expr.nodes[2] {
            Node::Expr(expr) => {
                check_num(&expr, 0, 3.0);
                check_num(&expr, 1, 4.0);
                check_bin_op(&expr, 0, 2, &bin_ops);
            }
            _ => assert!(false),
        };
        check_num(&expr, 0, 1.0);
        check_num(&expr, 1, 2.0);
        check_bin_op(&expr, 0, 0, &bin_ops);
    }

    #[test]
    fn test_make_parans_last_expression() {
        let (bin_ops, _) = make_binary_operators::<f32>();

        let expr_elts = vec![
            ExprElt::<f32>::Paran(Paran::Open),
            ExprElt::<f32>::Num(1.0),
            ExprElt::<f32>::BinOp(bin_ops[0].1),
            ExprElt::<f32>::Num(2.0),
            ExprElt::<f32>::Paran(Paran::Close),
            ExprElt::<f32>::BinOp(bin_ops[1].1),
            ExprElt::<f32>::Num(3.0),
            ExprElt::<f32>::BinOp(bin_ops[2].1),
            ExprElt::<f32>::Num(4.0),
        ];
        let (expr, _) = make_expression::<f32>(&expr_elts[0..]);
        match &expr.nodes[0] {
            Node::Expr(expr) => {
                check_num(&expr, 0, 1.0);
                check_num(&expr, 1, 2.0);
                check_bin_op(&expr, 0, 0, &bin_ops);
            }
            _ => assert!(false),
        };
        check_num(&expr, 1, 3.0);
        check_num(&expr, 2, 4.0);
        check_bin_op(&expr, 0, 1, &bin_ops);
        check_bin_op(&expr, 1, 2, &bin_ops);
    }

    #[test]
    fn test_ops() {
        fn check_add(op: fn(f32, f32) -> f32) {
            assert_eq!(op(1.0, 1.0), 2.0);
        }
        fn check_sub(op: fn(f32, f32) -> f32) {
            assert_eq!(op(1.0, 1.0), 0.0);
        }
        fn check_mul(op: fn(f32, f32) -> f32) {
            assert_eq!(op(1.0, 1.0), 1.0);
            assert_float_eq(op(7.5, 3.0), 22.5);
        }
        fn check_div(op: fn(f32, f32) -> f32) {
            assert_eq!(op(1.0, 1.0), 1.0);
            assert_float_eq(op(1.0, 2.0), 0.5);
        }

        let (bin_ops, _) = make_binary_operators::<f32>();
        check_add(find_op(&"+", &bin_ops).f);
        check_sub(find_op(&"-", &bin_ops).f);
        check_mul(find_op(&"*", &bin_ops).f);
        check_div(find_op(&"/", &bin_ops).f);
    }
    #[test]
    fn test_apply_regexes() {
        fn unpack(text: &str, reference: &Vec<ExprElt<f32>>) {
            let res = apply_regexes::<f32>(text);
            for (rs, rf) in izip!(res, reference) {
                match rs {
                    ExprElt::Num(n) => assert_float_eq(
                        n,
                        match rf {
                            ExprElt::Num(nf) => *nf,
                            _ => panic!("Ref wants Num"),
                        },
                    ),
                    _ => assert!(&rs == rf),
                }
            }
        }
        let (bin_ops, _) = make_binary_operators::<f32>();
        unpack("7.245", &vec![ExprElt::<f32>::Num(7.245)]);
        unpack("*", &vec![ExprElt::<f32>::BinOp(bin_ops[0].1)]);
        unpack(
            "2+7.245",
            &vec![
                ExprElt::<f32>::Num(2.0),
                ExprElt::<f32>::BinOp(bin_ops[2].1),
                ExprElt::<f32>::Num(7.245),
            ],
        );
        unpack(
            "1*2+(4-5)/2/3",
            &vec![
                ExprElt::<f32>::Num(1.0),
                ExprElt::<f32>::BinOp(bin_ops[0].1),
                ExprElt::<f32>::Num(2.0),
                ExprElt::<f32>::BinOp(bin_ops[2].1),
                ExprElt::<f32>::Paran(Paran::Open),
                ExprElt::<f32>::Num(4.0),
                ExprElt::<f32>::BinOp(bin_ops[3].1),
                ExprElt::<f32>::Num(5.0),
                ExprElt::<f32>::Paran(Paran::Close),
                ExprElt::<f32>::BinOp(bin_ops[1].1),
                ExprElt::<f32>::Num(2.0),
                ExprElt::<f32>::BinOp(bin_ops[1].1),
                ExprElt::<f32>::Num(3.0),
            ],
        );
    }
}
