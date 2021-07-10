//! Exexpress is an extendable expression evaluator for mathematical expressions.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exexpress::{eval_str};
//! assert!((eval_str("1.5 * ((cos(0) + 23.0) / 2.0)")? - 18.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! For floats, we have a list of predifined operators, namely
//! `^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log`, and `log2`. These are
//! defined in [`make_default_operators`](make_default_operators).
//!
//! ## Variables
//! For variables we can use curly brackets as shown in the following expression.
//! Variables' values are passed as slices to [`eval_expr`](eval_expr).
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exexpress::{eval_expr, make_default_operators, parse};
//! let to_be_parsed = "log({x}) + 2* (-{x}^2 + sin(4*{y}))";
//! let expr = parse::<f64>(to_be_parsed, make_default_operators::<f64>())?;
//! assert!((eval_expr::<f64>(&expr, &[2.5, 3.7]) - 14.992794866624788 as f64).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The `n`-th number in the slice corresponds to the `n`-th variable. Thereby only the
//! first occurence of the variables is relevant. In this example, we have `x=2.5` and `y=3.7`.
//!
//! ## Extendability
//! Library users can also define a different set of operators as shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exexpress::{eval_expr, parse, BinOp, Operator};
//! let ops = vec![
//!     Operator {
//!         repr: "%",
//!         bin_op: Some(BinOp{op: |a: i32, b: i32| a % b, prio: 1}),
//!         unary_op: None,
//!     },
//!     Operator {
//!         repr: "/",
//!         bin_op: Some(BinOp{op: |a: i32, b: i32| a / b, prio: 1}),
//!         unary_op: None,
//!     },
//! ];
//! let to_be_parsed = "19 % 5 / 2";
//! let expr = parse::<i32>(to_be_parsed, ops)?;
//! assert_eq!(eval_expr::<i32>(&expr, &[1, 0]), 2);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ### Operators
//!
//! Operators are instances of the struct
//! [`Operator`](Operator) that has its representation in the field 
//! [`repr`](Operator::repr), a binary and a unary operator of
//! type [`Option<BinOp<T>>`](Operator::bin_op) and 
//! [`Option<fn(T) -> T>`](Operator::unary_op), respectively, as 
//! members. [`BinOp`](BinOp)
//! contains in addition to the operator [`op`](BinOp::op) of type `fn(T, T) -> T` an 
//! integer [`prio`](BinOp::prio). Operators
//! can be both, binary and unary such as `-` as defined in the list of default
//! operators. Note that we expect a unary operator to be always on the left of a
//! number.
//!
//! ### Data Types of Numbers
//!
//! You can use any type that implements [`Copy`](core::marker::Copy) and 
//! [`FromStr`](std::str::FromStr). In case you do not pass a number that matches the 
//! regex `r"\.?[0-9]+(\.[0-9]+)?"`, you have to pass a suitable regex and use the 
//! function [`parse_with_number_pattern`](parse::parse_with_number_pattern) instead of
//! [`parse`](parse::parse). Here is an example for `bool`.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exexpress::{eval_expr, parse_with_number_pattern, BinOp, Operator};
//! let ops = vec![
//!     Operator {
//!         repr: "&&",
//!         bin_op: Some(BinOp{op: |a: bool, b: bool| a && b, prio: 1}),
//!         unary_op: None,
//!     },
//!     Operator {
//!         repr: "||",
//!         bin_op: Some(BinOp{op: |a: bool, b: bool| a || b, prio: 1}),
//!         unary_op: None,
//!     },
//!     Operator {
//!         repr: "!",
//!         bin_op: None,
//!         unary_op: Some(|a: bool| !a),
//!     },
//! ];
//! let to_be_parsed = "!(true && false)";
//! let expr = parse_with_number_pattern::<bool>(to_be_parsed, ops, "(true|false)")?;
//! assert_eq!(eval_expr::<bool>(&expr, &[]), true);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Priorities and Parentheses
//! In Exexpress-land, unary operators always have higher priority than binary operators, e.g.,
//! `-2^2=4` instead of `-2^2=-4`. Moreover, we are not too strict regarding parentheses.
//! For instance `"---1"` will evalute to `-1`.
//! If you want to be on the safe side, we suggest using parentheses.
//!

mod expression;
mod operators;
mod parse;
mod util;

pub use expression::{eval_expr, Expression, Node};

pub use parse::{parse, parse_with_default_ops, parse_with_number_pattern, ExParseError};

pub use operators::{make_default_operators, BinOp, Operator};

/// Parses a string, evaluates a string, and returns the resulting number.
pub fn eval_str(text: &str) -> Result<f64, ExParseError> {
    let expr = parse_with_default_ops(text)?;
    Ok(eval_expr(&expr, &vec![]))
}

#[cfg(test)]
mod tests {

    use std::iter::once;

    use crate::{
        eval_str,
        expression::eval_expr,
        operators::{make_default_operators, BinOp, Operator},
        parse::{parse, parse_with_number_pattern},
        util::{assert_float_eq_f32, assert_float_eq_f64},
    };

    #[test]
    fn test_bool() {
        let ops = vec![
            Operator {
                repr: "&&",
                bin_op: Some(BinOp {
                    op: |a: bool, b: bool| a && b,
                    prio: 1,
                }),
                unary_op: None,
            },
            Operator {
                repr: "||",
                bin_op: Some(BinOp {
                    op: |a: bool, b: bool| a || b,
                    prio: 1,
                }),
                unary_op: None,
            },
            Operator {
                repr: "!",
                bin_op: None,
                unary_op: Some(|a: bool| !a),
            },
        ];
        let to_be_parsed = "!(true && false)";
        let expr = parse_with_number_pattern::<bool>(to_be_parsed, ops, "(true|false)").unwrap();
        assert_eq!(eval_expr::<bool>(&expr, &[]), true);
    }
    #[test]
    fn test_variables() {
        let operators = make_default_operators::<f32>();

        let to_be_parsed = "5*{x} + 4*{y} + 3*{x}";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[1.0, 0.0]), 8.0);

        let to_be_parsed = "5*{x} + 4*{y}";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[0.0, 1.0]), 4.0);

        let to_be_parsed = "5*{x} + 4*{y} + {x}^2";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[2.5, 3.7]), 33.55);
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[12.0, 9.3]), 241.2);

        let to_be_parsed = "2*(4*{x} + {y}^2)";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[2.0, 3.0]), 34.0);

        let to_be_parsed = "sin({myvar_25})";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[1.5707963267948966]), 1.0);

        let to_be_parsed = "((sin({myvar_25})))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[1.5707963267948966]), 1.0);

        let to_be_parsed = "(0 * {myvar_25} + cos({X}))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(
            eval_expr::<f32>(&expr, &[1.5707963267948966, 3.141592653589793]),
            -1.0,
        );

        let to_be_parsed = "(-{X}^2)";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[1.0]), 1.0);

        let to_be_parsed = "log({x}) + 2* (-{x}^2 + sin(4*{y}))";
        let expr = parse::<f32>(to_be_parsed, operators.clone()).unwrap();
        assert_float_eq_f32(eval_expr::<f32>(&expr, &[2.5, 3.7]), 14.992794866624788);

        let ops = vec![
            Operator {
                repr: "invert",
                bin_op: None,
                unary_op: Some(|a: f32| 1.0 / a),
            },
            Operator {
                repr: "sqrt",
                bin_op: None,
                unary_op: Some(|a: f32| a.sqrt()),
            },
        ];
        let expr = parse::<f32>("sqrt(invert({a}))", ops).unwrap();
        assert_float_eq_f32(eval_expr(&expr, &[0.25]), 2.0);
    }

    #[test]
    fn test_custom_ops() {
        let custom_ops = vec![
            Operator {
                repr: "**",
                bin_op: Some(BinOp {
                    op: |a: f32, b| a.powf(b),
                    prio: 2,
                }),
                unary_op: None,
            },
            Operator {
                repr: "*",
                bin_op: Some(BinOp {
                    op: |a, b| a * b,
                    prio: 1,
                }),
                unary_op: None,
            },
            Operator {
                repr: "invert",
                bin_op: None,
                unary_op: Some(|a: f32| 1.0 / a),
            },
        ];
        let expr = parse::<f32>("2**2*invert(3)", custom_ops).unwrap();
        let val = eval_expr::<f32>(&expr, &vec![]);
        assert_float_eq_f32(val, 4.0 / 3.0);

        let zero_mapper = Operator {
            repr: "zer0",
            bin_op: Some(BinOp {
                op: |_: f32, _| 0.0,
                prio: 2,
            }),
            unary_op: Some(|_| 0.0),
        };
        let extended_operators = make_default_operators::<f32>()
            .iter()
            .cloned()
            .chain(once(zero_mapper))
            .collect::<Vec<_>>();
        let expr = parse::<f32>("2^2*1/({berti}) + zer0(4)", extended_operators).unwrap();
        let val = eval_expr::<f32>(&expr, &[4.0]);
        assert_float_eq_f32(val, 1.0);
    }

    #[test]
    fn test_eval() {
        assert_float_eq_f64(eval_str(&"2*3^2").unwrap(), 18.0);
        assert_float_eq_f64(eval_str(&"-3^2").unwrap(), 9.0);
        assert_float_eq_f64(eval_str(&"11.3").unwrap(), 11.3);
        assert_float_eq_f64(eval_str(&"+11.3").unwrap(), 11.3);
        assert_float_eq_f64(eval_str(&"-11.3").unwrap(), -11.3);
        assert_float_eq_f64(eval_str(&"(-11.3)").unwrap(), -11.3);
        assert_float_eq_f64(eval_str(&"11.3+0.7").unwrap(), 12.0);
        assert_float_eq_f64(eval_str(&"31.3+0.7*2").unwrap(), 32.7);
        assert_float_eq_f64(eval_str(&"1.3+0.7*2-1").unwrap(), 1.7);
        assert_float_eq_f64(eval_str(&"1.3+0.7*2-1/10").unwrap(), 2.6);
        assert_float_eq_f64(eval_str(&"(1.3+0.7)*2-1/10").unwrap(), 3.9);
        assert_float_eq_f64(eval_str(&"1.3+(0.7*2)-1/10").unwrap(), 2.6);
        assert_float_eq_f64(eval_str(&"1.3+0.7*(2-1)/10").unwrap(), 1.37);
        assert_float_eq_f64(eval_str(&"1.3+0.7*(2-1/10)").unwrap(), 2.63);
        assert_float_eq_f64(eval_str(&"-1*(1.3+0.7*(2-1/10))").unwrap(), -2.63);
        assert_float_eq_f64(eval_str(&"-1*(1.3+(-0.7)*(2-1/10))").unwrap(), 0.03);
        assert_float_eq_f64(eval_str(&"-1*((1.3+0.7)*(2-1/10))").unwrap(), -3.8);
        assert_float_eq_f64(eval_str(&"sin(3.14159265358979)").unwrap(), 0.0);
        assert_float_eq_f64(eval_str(&"0-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq_f64(eval_str(&"-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq_f64(eval_str(&"3-(-1+sin(1.5707963267948966)*2)").unwrap(), 2.0);
        assert_float_eq_f64(
            eval_str(&"3-(-1+sin(cos(-3.14159265358979))*2)").unwrap(),
            5.6829419696157935,
        );
        assert_float_eq_f64(
            eval_str(&"-(-1+((-3.14159265358979)/5)*2)").unwrap(),
            2.256637061435916,
        );
        assert_float_eq_f64(
            eval_str(&"-(-1+(sin(-3.14159265358979)/5)*2)").unwrap(),
            1.0,
        );
        assert_float_eq_f64(
            eval_str(&"-(-1+sin(cos(-3.14159265358979)/5)*2)").unwrap(),
            1.3973386615901224,
        );
        assert_float_eq_f64(eval_str(&"-cos(3.14159265358979)").unwrap(), 1.0);
        assert_float_eq_f64(
            eval_str(&"1+sin(-cos(-3.14159265358979))").unwrap(),
            1.8414709848078965,
        );
        assert_float_eq_f64(
            eval_str(&"-1+sin(-cos(-3.14159265358979))").unwrap(),
            -0.1585290151921035,
        );
        assert_float_eq_f64(
            eval_str(&"-(-1+sin(-cos(-3.14159265358979)/5)*2)").unwrap(),
            0.6026613384098776,
        );
        assert_float_eq_f64(eval_str(&"sin(-(2))*2").unwrap(), -1.8185948536513634);
        assert_float_eq_f64(eval_str(&"sin(sin(2))*2").unwrap(), 1.5781446871457767);
        assert_float_eq_f64(eval_str(&"sin(-(sin(2)))*2").unwrap(), -1.5781446871457767);
        assert_float_eq_f64(eval_str(&"-sin(2)*2").unwrap(), -1.8185948536513634);
        assert_float_eq_f64(eval_str(&"sin(-sin(2))*2").unwrap(), -1.5781446871457767);
        assert_float_eq_f64(eval_str(&"sin(-sin(2)^2)*2").unwrap(), 1.4715655294841483);
        assert_float_eq_f64(
            eval_str(&"sin(-sin(2)*-sin(2))*2").unwrap(),
            1.4715655294841483,
        );
        assert_float_eq_f64(eval_str(&"--(1)").unwrap(), 1.0);
        assert_float_eq_f64(eval_str(&"--1").unwrap(), 1.0);
        assert_float_eq_f64(eval_str(&"----1").unwrap(), 1.0);
        assert_float_eq_f64(eval_str(&"---1").unwrap(), -1.0);
        assert_float_eq_f64(eval_str(&"3-(4-2/3+(1-2*2))").unwrap(), 2.666666666666666);
        assert_float_eq_f64(
            eval_str(&"log(log(2))*tan(2)+exp(1.5)").unwrap(),
            5.2825344122094045,
        );
        assert_float_eq_f64(
            eval_str(&"log(log2(2))*tan(2)+exp(1.5)").unwrap(),
            4.4816890703380645,
        );
        assert_float_eq_f64(eval_str(&"log2(2)").unwrap(), 1.0);
    }

    #[test]
    fn test_error_handling() {
        assert!(eval_str(&"").is_err());
        assert!(eval_str(&"5+5-(").is_err());
        assert!(eval_str(&")2*(5+5)*3-2)*2").is_err());
        assert!(eval_str(&"2*(5+5))").is_err());
    }
}
