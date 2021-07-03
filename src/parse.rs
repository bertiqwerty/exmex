use std::str::FromStr;

use num::Float;
use regex::{Regex, RegexSet};

use crate::types::{BinaryOperator, Expression, Node};

type VecBinOps<'a, T> = Vec<(&'a str, BinaryOperator<T>)>;

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
                    panic!("Internal error. Paran {} is neither ( not ). Check the paran-regex.", c);
                })
            } else if matches.matched(2) {
                ExprElt::<T>::BinOp(find_op(elt_str, &bin_ops))
            } else {
                panic!("Internal regex mismatch!");
            }
        })
        .collect()
}

fn make_expression<T: Float + FromStr + std::fmt::Debug>(expr_elts: &[ExprElt<T>]) -> (Expression<T>, usize) {
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
            },
            ExprElt::Num(n) => {
                result.nodes.push(Node::Num(n));
                i += 1;
            },
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
                },
            },
        }
    }
    (result, i)
}

pub fn parse<T: Float + FromStr + std::fmt::Debug>(text: &str) -> Expression<T>
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let elts = apply_regexes::<T>(text);
    let (expr, _) = make_expression(&elts[0..]);
    expr
}

#[cfg(test)]
mod tests {
    use itertools::izip;

    use crate::{parse::{
            apply_regexes, find_op, make_binary_operators, make_expression, ExprElt, Expression,
            Node, Paran,
        }, types::BinaryOperator, util::assert_float_eq};

    fn check_num(expr: &Expression<f32>, idx: usize, reference: f32) {
        match expr.nodes[idx] {
            Node::<f32>::Num(n) => assert_float_eq(n, reference),
            _ => assert!(false),
        }
    }
    fn check_bin_op(expr: &Expression<f32>, idx: usize, reference: usize, bin_ops: &Vec<(&str, BinaryOperator<f32>)>) {
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
