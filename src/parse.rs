use std::str::FromStr;

use num::Float;
use regex::Regex;

use crate::types::{BinaryOperator, Expression, Node};

fn find_numbers<T: Float + FromStr>(text: &str) -> Result<Vec<T>, <T as FromStr>::Err> {
    let re = Regex::new("\\.?[0-9]+(\\.[0-9]+)?").unwrap();
    re.captures_iter(text).map(|c| c[0].parse::<T>()).collect()
}

type VecBinOps<'a, T> = Vec<(&'a str, BinaryOperator<T>)>;

fn make_binary_operators<'a, T: Float>() -> (VecBinOps<'a, T>, Regex) {
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
        Regex::new("[*/+\\-]").unwrap(),
    )
}

fn find_op<'a, T: Float>(name: &str, ops: &VecBinOps<'a, T>) -> BinaryOperator<T> {
    ops.iter().find(|(op_name, _)| op_name == &name).unwrap().1
}

fn find_operators<T: Float>(text: &str) -> Vec<BinaryOperator<T>> {
    let (bin_ops, bin_re) = make_binary_operators::<T>();
    bin_re
        .captures_iter(text)
        .map(|c| find_op(&c[0], &bin_ops))
        .collect()
}

pub fn parse_flat_exp<T: Float + FromStr>(
    text: &str,
) -> Result<Expression<T>, <T as FromStr>::Err> {
    Ok(Expression {
        bin_ops: find_operators::<T>(text),
        nodes: find_numbers(text)?.iter().map(|n| Node::NUM(*n)).collect(),
    })
}

#[derive(Debug, PartialEq, Eq)]
enum Paran {
    OPEN(usize),
    CLOSE(usize),
}

fn find_parantheses(text: &str) -> Vec<Paran> {
    text.chars()
        .enumerate()
        .filter(|(_, c)| *c == '(' || *c == ')')
        .map(|(i, c)| {
            if c == '(' {
                Paran::OPEN(i)
            } else {
                Paran::CLOSE(i)
            }
        })
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use crate::{
        parse::{
            find_numbers, find_op, find_operators, find_parantheses, make_binary_operators, Paran,
        },
        util::assert_float_eq,
    };

    #[test]
    fn test_find_numbers() {
        assert_eq!(find_numbers::<f32>("3.4").unwrap(), vec![3.4]);
        assert_eq!(find_numbers::<f32>("3.465  ").unwrap(), vec![3.465]);
        assert_eq!(find_numbers::<f32>("0.524 +3.4").unwrap(), vec![0.524, 3.4]);
        assert_eq!(
            find_numbers::<f32>("00.524+ 3.4").unwrap(),
            vec![0.524, 3.4]
        );
        assert_eq!(find_numbers::<f32>("0 + 0").unwrap(), vec![0.0, 0.0]);
        assert_eq!(
            find_numbers::<f32>("0.0.0+0.").unwrap(),
            vec![0.0, 0.0, 0.0]
        );
        assert_eq!(find_numbers::<f32>(".5+.0").unwrap(), vec![0.5, 0.0]);
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

        fn check_ops(text: &str) {
            let ops = find_operators::<f32>(text);
            check_add(ops[0].f);
            check_sub(ops[1].f);
            check_mul(ops[2].f);
            check_div(ops[3].f);
        }
        check_ops(&"4.4 + 5-6 *  7/8");
        check_ops(&"0+.7- 32.623*733.1/.8");
        check_ops(&".4+. .5-6..*7./8.");
    }

    #[test]
    fn test_find_parans() {
        assert_eq!(
            find_parantheses("()"),
            vec![Paran::OPEN(0), Paran::CLOSE(1)]
        );
        assert_eq!(find_parantheses(""), vec![]);
        assert_eq!(
            find_parantheses("4/(3*(4+5)) * (4-5)"),
            vec![
                Paran::OPEN(0),
                Paran::OPEN(3),
                Paran::CLOSE(7),
                Paran::CLOSE(8),
                Paran::OPEN(12),
                Paran::CLOSE(16)
            ]
        );
    }
}
