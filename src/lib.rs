#![doc(html_root_url = "https://docs.rs/exmex/0.9.4")]
//! Exmex is a fast, simple, and **ex**tendable **m**athematical **ex**pression evaluator
//! with the ability to compute [partial derivatives](FlatEx::partial) of expressions.  
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex;
//! assert!((exmex::eval_str::<f64>("1.5 * ((cos(0) + 23.0) / 2.0)")? - 18.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! For floats, we have a list of predifined operators containing
//! `^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log`, and `log2`. The full list is
//! defined in [`DefaultOpsFactory`](DefaultOpsFactory). Library users can also create their
//! own operators as shown below in the section about extendability.
//!
//! ## Variables
//! For variables we can use strings that are not in the list of operators as shown in the following expression.
//! Additionally, variables should consist only of letters, numbers, and underscores. More precisely, they need to fit the
//! regular expression
//! ```r"^[a-zA-Z_]+[a-zA-Z_0-9]*"```.
//! Variables' values are passed as slices to [`eval`](Express::eval).
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let to_be_parsed = "log(z) + 2* (-z^2 + sin(4*y))";
//! let expr = exmex::parse::<f64>(to_be_parsed)?;
//! assert!((expr.eval(&[3.7, 2.5])? - 14.992794866624788 as f64).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The `n`-th number in the slice corresponds to the `n`-th variable. Thereby, the
//! alphatical order of the variables is relevant. In this example, we have `y=3.7` and `z=2.5`.
//! If variables are between curly brackets, they can have arbitrary names, e.g.,
//! `{456/549*(}`, `{x}`, and confusingly even `{x+y}` are valid variable names as shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let x = 2.1f64;
//! let y = 0.1f64;
//! let to_be_parsed = "log({x+y})";  // {x+y} is the name of one(!) variable 😕.
//! let expr = exmex::parse::<f64>(to_be_parsed)?;
//! assert!((expr.eval(&[x+y])? - 2.2f64.ln()).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The value returned by [`parse`](parse) implements the [`Express`](Express) trait
//! and is an instance of the struct [`FlatEx`](FlatEx).
//! ## Extendability
//! Library users can define their own operator factory as shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{BinOp, MakeOperators, Operator, ops_factory};
//! ops_factory!(
//!     IntegerOpsFactory,  // name of the factory type
//!     i32,                // data type of the operands
//!     Operator {
//!         repr: "%",
//!         bin_op: Some(BinOp{ apply: |a, b| a % b, prio: 1 }),
//!         unary_op: None,
//!         },
//!     Operator {
//!         repr: "/",
//!         bin_op: Some(BinOp{ apply: |a, b| a / b, prio: 1 }),
//!         unary_op: None,
//!     }
//! );
//! let to_be_parsed = "19 % 5 / 2 / a";
//! let expr = FlatEx::<_, IntegerOpsFactory>::from_str(to_be_parsed)?;
//! assert_eq!(expr.eval(&[1])?, 2);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! One does not have to use the `ops_factory` macro, e.g., if one wants to 
//! extend an existing list of operators. In this case one has to create a factory 
//! `struct` and implement the [`MakeOperators`](MakeOperators) trait.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{BinOp, DefaultOpsFactory, MakeOperators, Operator};
//! #[derive(Clone)]
//! struct ExtendedOpsFactory;
//! impl MakeOperators<f32> for ExtendedOpsFactory {
//!     fn make<'a>() -> Vec<Operator<'a, f32>> {
//!         let mut ops = DefaultOpsFactory::<f32>::make();
//!         ops.push(
//!             Operator {
//!                 repr: "invert",
//!                 bin_op: None,
//!                 unary_op: Some(|a: f32| 1.0 / a),
//!             },
//!         );
//!         ops
//!     }
//! }
//! let to_be_parsed = "1 / a + invert(a)";
//! let expr = FlatEx::<_, ExtendedOpsFactory>::from_str(to_be_parsed)?;
//! assert!((expr.eval(&[3.0])? - 2.0/3.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! ### Operators
//!
//! Operators are instances of the struct
//! [`Operator`](Operator) that has its representation in the field
//! [`repr`](Operator::repr), a binary and a unary operator of
//! type [`Option<BinOp<T>>`](Operator::bin_op) and
//! [`Option<fn(T) -> T>`](Operator::unary_op), respectively, as
//! members. [`BinOp`](BinOp)
//! contains in addition to the function pointer [`apply`](BinOp::apply) of type `fn(T, T) -> T` an
//! integer [`prio`](BinOp::prio). Operators
//! can be both, binary and unary. See, e.g.,  `-` defined in the list of default
//! operators. Note that we expect a unary operator to be always on the left of a
//! number.
//!
//! ### Data Types of Numbers
//!
//! You can use any type that implements [`Copy`](core::marker::Copy) and
//! [`FromStr`](std::str::FromStr). In case the representation of your data type in the
//! string does not match the number regex `r"\.?[0-9]+(\.[0-9]+)?"`, you have to pass a
//! suitable regex and use the function
//! [`from_pattern`](Express::from_pattern) instead of
//! [`from_str`](Express::from_str). Here is an example for `bool`.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{BinOp, MakeOperators, Operator, ops_factory};
//! ops_factory!(
//!     BooleanOpsFactory, 
//!     bool,
//!     Operator {
//!         repr: "&&",
//!         bin_op: Some(BinOp{ apply: |a, b| a && b, prio: 1 }),
//!         unary_op: None,
//!     },
//!     Operator {
//!         repr: "||",
//!         bin_op: Some(BinOp{ apply: |a, b| a || b, prio: 1 }),
//!         unary_op: None,
//!     },
//!     Operator {
//!         repr: "!",
//!         bin_op: None,
//!         unary_op: Some(|a| !a),
//!     }
//! );
//! let to_be_parsed = "!(true && false) || (!false || (true && false))";
//! let expr = FlatEx::<_, BooleanOpsFactory>::from_pattern(to_be_parsed, "true|false")?;
//! assert_eq!(expr.eval(&[])?, true);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Partial Derivatives
//!
//! For default operators, expressions can be transformed into their partial derivatives
//! again represented by expressions.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let expr = exmex::parse::<f64>("x^2 + y^2")?;
//! let dexpr_dx = expr.clone().partial(0)?;
//! let dexpr_dy = expr.partial(1)?;
//! assert!((dexpr_dx.eval(&[3.0, 2.0])? - 6.0).abs() < 1e-12);
//! assert!((dexpr_dy.eval(&[3.0, 2.0])? - 4.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Owned Expression
//! You cannot return all expression types from a function without a lifetime parameter.
//! For instance, expressions that are instances of [`FlatEx`](FlatEx) keep `&str`s instead of
//! `String`s of variable or operator names to make faster parsing possible.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::ExResult;
//! fn create_expr<'a>() -> ExResult<FlatEx::<'a, f64>> {
//! //       |                          |
//! //       lifetime parameter necessary
//!
//!     let to_be_parsed = "log(z) + 2* (-z^2 + sin(4*y))";
//!     exmex::parse::<f64>(to_be_parsed)
//! }
//! let expr = create_expr()?;
//! assert!((expr.eval(&[3.7, 2.5])? - 14.992794866624788 as f64).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! If you are willing to pay the price of roughly doubled parsing times, you can
//! obtain an expression that is an instance of [`OwnedFlatEx`](OwnedFlatEx) and owns
//! its strings. Evaluation times should be comparable. However, a lifetime parameter is
//! not needed anymore as shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::{ExResult, Express, OwnedFlatEx};
//! fn create_expr() -> ExResult<OwnedFlatEx::<f64>> {
//!     let to_be_parsed = "log(z) + 2* (-z^2 + sin(4*y))";
//!     OwnedFlatEx::<f64>::from_str(to_be_parsed)
//! }
//! let expr_owned = create_expr()?;
//! assert!((expr_owned.eval(&[3.7, 2.5])? - 14.992794866624788 as f64).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Priorities and Parentheses
//! In Exmex-land, unary operators always have higher priority than binary operators, e.g.,
//! `-2^2=4` instead of `-2^2=-4`. Moreover, we are not too strict regarding parentheses.
//! For instance
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex;
//! assert_eq!(exmex::eval_str::<f64>("---1")?, -1.0);
//! #
//! #     Ok(())
//! # }
//! ```
//! If you want to be on the safe side, we suggest using parentheses.
//!
//! ## Display
//!
//! An instance of [`FlatEx`](FlatEx) can be displayed as string. Note that this
//! [`unparse`](FlatEx::unparse)d string does not necessarily coincide with the original
//! string, since, e.g., curly brackets are added and expressions are compiled.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let expr = exmex::parse::<f64>("-sin(z)/cos(mother_of_names) + 2^7")?;
//! assert_eq!(format!("{}", expr), "-(sin({z}))/cos({mother_of_names})+128.0");
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Serialization and Deserialization
//!
//! To use [`serde`](https://serde.rs/) you can activate the feature `serde`.
//! The implementation un-parses and re-parses the whole expression.
//! [`Deserialize`](https://docs.serde.rs/serde/de/trait.Deserialize.html) and
//! [`Serialize`](https://docs.serde.rs/serde/de/trait.Serialize.html) are implemented for
//! both, [`FlatEx`](FlatEx) and [`OwnedFlatEx`](OwnedFlatEx).
//!
//! ## Unicode
//! Unicode input strings are currently not supported 😕 but might be added in the
//! future 😀.
//!

use std::{fmt::Debug, str::FromStr};

use expression::deep::DeepEx;
use num::Float;

mod definitions;
mod expression;
#[macro_use]
mod operators;
mod parser;
mod result;
mod util;
pub use {
    expression::{
        flat::{FlatEx, OwnedFlatEx},
        Express,
    },
    operators::{BinOp, DefaultOpsFactory, MakeOperators, Operator},
    result::{ExError, ExResult},
};

/// To use the expression trait [`Express`](Express) and its implementation [`FlatEx`](FlatEx)
/// one can `use exmex::prelude::*;`.
pub mod prelude {
    pub use super::expression::{flat::FlatEx, Express};
}

/// Parses a string, evaluates a string, and returns the resulting number.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`](ExError) is returned.
///
pub fn eval_str<T: Float + FromStr + Debug>(text: &str) -> ExResult<T>
where
    <T as FromStr>::Err: Debug,
{
    let deepex = DeepEx::<T>::from_str_float(text)?;
    deepex.eval(&[])
}

/// Parses a string and returns the expression that can be evaluated.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`](ExError) is returned.
///
pub fn parse<T: Float + FromStr + Debug>(text: &str) -> ExResult<FlatEx<T>>
where
    <T as FromStr>::Err: Debug,
{
    FlatEx::<T>::from_str(text)
}

#[cfg(test)]
mod tests {

    use std::{iter::once, ops::Range};

    use rand::Rng;
    use smallvec::{smallvec, SmallVec};

    use crate::prelude::*;
    use crate::{
        eval_str,
        operators::{BinOp, DefaultOpsFactory, MakeOperators, Operator},
        parse,
        util::{assert_float_eq_f32, assert_float_eq_f64},
        ExResult, OwnedFlatEx,
    };

    #[test]
    fn test_readme() {
        fn readme_partial() -> ExResult<()> {
            let expr = parse::<f64>("y*x^2")?;

            // d_x
            let dexpr_dx = expr.partial(0)?;
            assert_eq!(format!("{}", dexpr_dx), "({x}*2.0)*{y}");

            // d_xy
            let ddexpr_dxy = dexpr_dx.partial(1)?;
            assert_eq!(format!("{}", ddexpr_dxy), "{x}*2.0");
            assert_float_eq_f64(ddexpr_dxy.eval(&[2.0, f64::MAX])?, 4.0);

            // d_xyx
            let dddexpr_dxyx = ddexpr_dxy.partial(0)?;
            assert_eq!(format!("{}", dddexpr_dxyx), "2.0");
            assert_float_eq_f64(dddexpr_dxyx.eval(&[f64::MAX, f64::MAX])?, 2.0);

            Ok(())
        }
        fn readme() -> ExResult<()> {
            let result = eval_str("sin(73)")?;
            assert_float_eq_f64(result, 73f64.sin());
            let expr = parse::<f64>("2*x^3-4/z")?;
            let value = expr.eval(&[5.3, 0.5])?;
            assert_float_eq_f64(value, 289.75399999999996);
            Ok(())
        }
        fn readme_int() -> ExResult<()> {
            ops_factory!(
                BitwiseOpsFactory,
                u32,
                Operator {
                    repr: "|",
                    bin_op: Some(BinOp {
                        apply: |a, b| a | b,
                        prio: 0,
                    }),
                    unary_op: None,
                },
                Operator {
                    repr: "!",
                    bin_op: None,
                    unary_op: Some(|a| !a),
                }
            );
            let expr = FlatEx::<_, BitwiseOpsFactory>::from_str("!(a|b)")?;
            let result = expr.eval(&[0, 1])?;
            assert_eq!(result, u32::MAX - 1);
            Ok(())
        }
        assert!(!readme_partial().is_err());
        assert!(!readme().is_err());
        assert!(!readme_int().is_err());
    }
    #[test]
    fn test_variables_curly_space_names() {
        let sut = "{x } + { y }";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.0, 1.0]).unwrap(), 2.0);
        assert_eq!(expr.unparse().unwrap(), "{x }+{ y }");
        let sut = "2*(4*{ xasd sa } + { y z}^2)";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[2.0, 3.0]).unwrap(), 34.0);
        assert_eq!(expr.unparse().unwrap(), "2.0*(4.0*{ xasd sa }+{ y z}^2.0)");
    }
    #[test]
    fn test_variables_curly() {
        let sut = "5*{x} +  4*log2(log(1.5+{gamma}))*({x}*-(tan(cos(sin(652.2-{gamma}))))) + 3*{x}";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.2, 1.0]).unwrap(), 8.040556934857268);

        let sut = "sin({myvwmlf4i58eo;w/-sin(a)r_25})";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.5707963267948966]).unwrap(), 1.0);

        let sut = "((sin({myvar_25})))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.5707963267948966]).unwrap(), 1.0);
    }
    #[test]
    fn test_variables() {
        let sut = "sin  ({x})+(((cos({y})   ^  (sin({z})))*log(cos({y})))*cos({z}))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        let reference =
            |x: f64, y: f64, z: f64| x.sin() + y.cos().powf(z.sin()) * y.cos().ln() * z.cos();

        assert_float_eq_f64(
            expr.eval(&[-0.18961918881278095, -6.383306547710852, 3.1742139703464503])
                .unwrap(),
            reference(-0.18961918881278095, -6.383306547710852, 3.1742139703464503),
        );

        let sut = "sin(sin(x - 1 / sin(y * 5)) + (5.0 - 1/z))";
        let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
        let reference =
            |x: f64, y: f64, z: f64| ((x - 1.0 / (y * 5.0).sin()).sin() + (5.0 - 1.0 / z)).sin();
        assert_float_eq_f64(
            expr.eval(&[1.0, 2.0, 4.0]).unwrap(),
            reference(1.0, 2.0, 4.0),
        );

        let sut = "0.02*sin( - (3*(2*(5.0 - 1/z))))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |z: f64| 0.02 * (-(3.0 * (2.0 * (5.0 - 1.0 / z)))).sin();
        assert_float_eq_f64(expr.eval(&[4.0]).unwrap(), reference(4.0));

        let sut = "y + 1 + 0.5 * x";
        let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[3.0, 1.0]).unwrap(), 3.5);

        let sut = " -(-(1+x))";
        let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 2.0);

        let sut = " sin(cos(-3.14159265358979*x))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), -0.841470984807896);

        let sut = "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)";
        let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.5, 0.2532]).unwrap(), -3.1164569260604176);

        let sut = "5*x + 4*y + 3*x";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.0, 0.0]).unwrap(), 8.0);

        let sut = "5*x + 4*y";
        let expr = OwnedFlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[0.0, 1.0]).unwrap(), 4.0);

        let sut = "5*x + 4*y + x^2";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 33.55);
        assert_float_eq_f64(expr.eval(&[12.0, 9.3]).unwrap(), 241.2);

        let sut = "2*(4*x + y^2)";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[2.0, 3.0]).unwrap(), 34.0);

        let sut = "sin(myvar_25)";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.5707963267948966]).unwrap(), 1.0);

        let sut = "((sin(myvar_25)))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.5707963267948966]).unwrap(), 1.0);

        let sut = "(0 * myvar_25 + cos(x))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(
            expr.eval(&[1.5707963267948966, 3.141592653589793]).unwrap(),
            -1.0,
        );

        let sut = "(-x^2)";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[1.0]).unwrap(), 1.0);

        let sut = "log(x) + 2* (-x^2 + sin(4*y))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[2.5, 3.7]).unwrap(), 14.992794866624788);

        let sut = "-sqrt(x)/(tanh(5-x)*2) + floor(2.4)* 1/asin(-x^2 + sin(4*sinh(y)))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(
            expr.eval(&[2.5, 3.7]).unwrap(),
            -(2.5f64.sqrt()) / (2.5f64.tanh() * 2.0)
                + 2.0 / ((3.7f64.sinh() * 4.0).sin() + 2.5 * 2.5).asin(),
        );

        let sut = "asin(sin(x)) + acos(cos(x)) + atan(tan(x))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[0.5]).unwrap(), 1.5);

        let sut = "sqrt(alpha^ceil(centauri))";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[2.0, 3.1]).unwrap(), 4.0);

        let sut = "trunc(x) + fract(x)";
        let expr = FlatEx::<f64>::from_str(sut).unwrap();
        assert_float_eq_f64(expr.eval(&[23422.52345]).unwrap(), 23422.52345);
    }

    #[test]
    fn test_custom_ops_invert() {
        #[derive(Clone)]
        struct SomeF32Operators;
        impl MakeOperators<f32> for SomeF32Operators {
            fn make<'a>() -> Vec<Operator<'a, f32>> {
                vec![
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
                ]
            }
        }
        let expr = OwnedFlatEx::<f32, SomeF32Operators>::from_str("sqrt(invert(a))").unwrap();
        assert_float_eq_f32(expr.eval(&[0.25]).unwrap(), 2.0);
    }

    #[test]
    fn test_custom_ops() {
        #[derive(Clone)]
        struct SomeF32Operators;
        impl MakeOperators<f32> for SomeF32Operators {
            fn make<'a>() -> Vec<Operator<'a, f32>> {
                vec![
                    Operator {
                        repr: "**",
                        bin_op: Some(BinOp {
                            apply: |a: f32, b| a.powf(b),
                            prio: 2,
                        }),
                        unary_op: None,
                    },
                    Operator {
                        repr: "*",
                        bin_op: Some(BinOp {
                            apply: |a, b| a * b,
                            prio: 1,
                        }),
                        unary_op: None,
                    },
                    Operator {
                        repr: "invert",
                        bin_op: None,
                        unary_op: Some(|a: f32| 1.0 / a),
                    },
                    Operator {
                        repr: "None",
                        bin_op: None,
                        unary_op: None,
                    },
                ]
            }
        }
        let expr = OwnedFlatEx::<f32, SomeF32Operators>::from_str("2**2*invert(3)").unwrap();
        let val = expr.eval(&[]).unwrap();
        assert_float_eq_f32(val, 4.0 / 3.0);

        let expr = FlatEx::<f32, SomeF32Operators>::from_str("None(5)");
        assert!(expr.is_err());

        #[derive(Clone)]
        struct ExtendedF32Operators;
        impl MakeOperators<f32> for ExtendedF32Operators {
            fn make<'a>() -> Vec<Operator<'a, f32>> {
                let zero_mapper = Operator {
                    repr: "zer0",
                    bin_op: Some(BinOp {
                        apply: |_: f32, _| 0.0,
                        prio: 2,
                    }),
                    unary_op: Some(|_| 0.0),
                };
                DefaultOpsFactory::<f32>::make()
                    .iter()
                    .cloned()
                    .chain(once(zero_mapper))
                    .collect::<Vec<_>>()
            }
        }
        let expr =
            FlatEx::<f32, ExtendedF32Operators>::from_str("2^2*1/(berti) + zer0(4)").unwrap();
        let val = expr.eval(&[4.0]).unwrap();
        assert_float_eq_f32(val, 1.0);
    }

    #[test]
    fn test_partial() {
        fn test(
            var_idx: usize,
            n_vars: usize,
            random_range: Range<f64>,
            flatex: FlatEx<f64>,
            reference: fn(f64) -> f64,
        ) {
            let deri = flatex.clone().partial(var_idx).unwrap();
            println!("flatex {}", flatex);
            println!("partial {}", deri);
            let mut rng = rand::thread_rng();
            for _ in 0..3 {
                let vut = rng.gen_range(random_range.clone());
                let mut vars: SmallVec<[f64; 10]> = smallvec![0.0; n_vars];
                vars[var_idx] = vut;
                println!("value under test {}.", vut);
                assert_float_eq_f64(deri.eval(&vars).unwrap(), reference(vut));
            }

            let owned_flatex = OwnedFlatEx::from_flatex(flatex);
            println!("flatex owned {}", owned_flatex);
            let owned_deri = owned_flatex.partial(var_idx).unwrap();
            println!("partial owned {}", owned_deri);
            for _ in 0..3 {
                let vut = rng.gen_range(random_range.clone());
                let mut vars: SmallVec<[f64; 10]> = smallvec![0.0; n_vars];
                vars[var_idx] = vut;
                println!("value under test {}.", vut);
                assert_float_eq_f64(owned_deri.eval(&vars).unwrap(), reference(vut));
            }
        }

        let sut = "+x";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |_: f64| 1.0;
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "++x";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |_: f64| 1.0;
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "+-+x";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |_: f64| -1.0;
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "-x";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |_: f64| -1.0;
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "--x";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |_: f64| 1.0;
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "sin(sin(x))";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |x: f64| x.sin().cos() * x.cos();
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "sin(x)-cos(x)+a";
        println!("{}", sut);
        let var_idx = 1;
        let n_vars = 2;
        let flatex_1 = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |x: f64| x.cos() + x.sin();
        test(
            var_idx,
            n_vars,
            -10000.0..10000.0,
            flatex_1.clone(),
            reference,
        );
        let deri = flatex_1.partial(var_idx).unwrap();
        let reference = |x: f64| -x.sin() + x.cos();
        test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);
        let deri = deri.partial(var_idx).unwrap();
        let reference = |x: f64| -x.cos() - x.sin();
        test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);
        let deri = deri.partial(var_idx).unwrap();
        let reference = |x: f64| x.sin() - x.cos();
        test(var_idx, n_vars, -10000.0..10000.0, deri.clone(), reference);

        let sut = "sin(x)-cos(x)+tan(x)+a";
        println!("{}", sut);
        let var_idx = 1;
        let n_vars = 2;
        let flatex_1 = FlatEx::<f64>::from_str("sin(x)-cos(x)+tan(x)+a").unwrap();
        let reference = |x: f64| x.cos() + x.sin() + 1.0 / (x.cos().powf(2.0));
        test(var_idx, n_vars, -10000.0..10000.0, flatex_1, reference);

        let sut = "log(v)*exp(v)+cos(x)+tan(x)+a";
        println!("{}", sut);
        let var_idx = 1;
        let n_vars = 3;
        let flatex = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |x: f64| 1.0 / x * x.exp() + x.ln() * x.exp();
        test(var_idx, n_vars, 0.01..100.0, flatex, reference);

        let sut = "a+z+sinh(v)/cosh(v)+b+tanh({v})";
        println!("{}", sut);
        let var_idx = 2;
        let n_vars = 4;
        let flatex = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |x: f64| {
            (x.cosh() * x.cosh() - x.sinh() * x.sinh()) / x.cosh().powf(2.0)
                + 1.0 / (x.cosh().powf(2.0))
        };
        test(var_idx, n_vars, -100.0..100.0, flatex, reference);

        let sut = "w+z+acos(v)+asin(v)+b+atan({v})";
        println!("{}", sut);
        let var_idx = 1;
        let n_vars = 4;
        let flatex = FlatEx::<f64>::from_str(sut).unwrap();
        let reference = |x: f64| {
            1.0 / (1.0 - x.powf(2.0)).sqrt() - 1.0 / (1.0 - x.powf(2.0)).sqrt()
                + 1.0 / (1.0 + x.powf(2.0))
        };
        test(var_idx, n_vars, -1.0..1.0, flatex, reference);

        let sut = "sqrt(var)*var^1.57";
        println!("{}", sut);
        let var_idx = 0;
        let n_vars = 1;
        let flatex = FlatEx::<f64>::from_str(sut).unwrap();
        let reference =
            |x: f64| 1.0 / (2.0 * x.sqrt()) * x.powf(1.57) + x.sqrt() * 1.57 * x.powf(0.57);
        test(var_idx, n_vars, 0.0..100.0, flatex, reference);
    }

    #[test]
    fn test_eval() {
        assert_float_eq_f64(eval_str("2*3^2").unwrap(), 18.0);
        assert_float_eq_f64(eval_str("-3^2").unwrap(), 9.0);
        assert_float_eq_f64(eval_str("11.3").unwrap(), 11.3);
        assert_float_eq_f64(eval_str("+11.3").unwrap(), 11.3);
        assert_float_eq_f64(eval_str("-11.3").unwrap(), -11.3);
        assert_float_eq_f64(eval_str("(-11.3)").unwrap(), -11.3);
        assert_float_eq_f64(eval_str("11.3+0.7").unwrap(), 12.0);
        assert_float_eq_f64(eval_str("31.3+0.7*2").unwrap(), 32.7);
        assert_float_eq_f64(eval_str("1.3+0.7*2-1").unwrap(), 1.7);
        assert_float_eq_f64(eval_str("1.3+0.7*2-1/10").unwrap(), 2.6);
        assert_float_eq_f64(eval_str("(1.3+0.7)*2-1/10").unwrap(), 3.9);
        assert_float_eq_f64(eval_str("1.3+(0.7*2)-1/10").unwrap(), 2.6);
        assert_float_eq_f64(eval_str("1.3+0.7*(2-1)/10").unwrap(), 1.37);
        assert_float_eq_f64(eval_str("1.3+0.7*(2-1/10)").unwrap(), 2.63);
        assert_float_eq_f64(eval_str("-1*(1.3+0.7*(2-1/10))").unwrap(), -2.63);
        assert_float_eq_f64(eval_str("-1*(1.3+(-0.7)*(2-1/10))").unwrap(), 0.03);
        assert_float_eq_f64(eval_str("-1*((1.3+0.7)*(2-1/10))").unwrap(), -3.8);
        assert_float_eq_f64(eval_str("sin(3.14159265358979)").unwrap(), 0.0);
        assert_float_eq_f64(eval_str("0-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq_f64(eval_str("-sin(3.14159265358979 / 2)").unwrap(), -1.0);
        assert_float_eq_f64(eval_str("3-(-1+sin(1.5707963267948966)*2)").unwrap(), 2.0);
        assert_float_eq_f64(
            eval_str("3-(-1+sin(cos(-3.14159265358979))*2)").unwrap(),
            5.6829419696157935,
        );
        assert_float_eq_f64(
            eval_str("-(-1+((-3.14159265358979)/5)*2)").unwrap(),
            2.256637061435916,
        );
        assert_float_eq_f64(eval_str("((2-4)/5)*2").unwrap(), -0.8);
        assert_float_eq_f64(eval_str("-(-1+(sin(-3.14159265358979)/5)*2)").unwrap(), 1.0);
        assert_float_eq_f64(
            eval_str("-(-1+sin(cos(-3.14159265358979)/5)*2)").unwrap(),
            1.3973386615901224,
        );
        assert_float_eq_f64(eval_str("-cos(3.14159265358979)").unwrap(), 1.0);
        assert_float_eq_f64(
            eval_str("1+sin(-cos(-3.14159265358979))").unwrap(),
            1.8414709848078965,
        );
        assert_float_eq_f64(
            eval_str("-1+sin(-cos(-3.14159265358979))").unwrap(),
            -0.1585290151921035,
        );
        assert_float_eq_f64(
            eval_str("-(-1+sin(-cos(-3.14159265358979)/5)*2)").unwrap(),
            0.6026613384098776,
        );
        assert_float_eq_f64(eval_str("sin(-(2))*2").unwrap(), -1.8185948536513634);
        assert_float_eq_f64(eval_str("sin(sin(2))*2").unwrap(), 1.5781446871457767);
        assert_float_eq_f64(eval_str("sin(-(sin(2)))*2").unwrap(), -1.5781446871457767);
        assert_float_eq_f64(eval_str("-sin(2)*2").unwrap(), -1.8185948536513634);
        assert_float_eq_f64(eval_str("sin(-sin(2))*2").unwrap(), -1.5781446871457767);
        assert_float_eq_f64(eval_str("sin(-sin(2)^2)*2").unwrap(), 1.4715655294841483);
        assert_float_eq_f64(
            eval_str("sin(-sin(2)*-sin(2))*2").unwrap(),
            1.4715655294841483,
        );
        assert_float_eq_f64(eval_str("--(1)").unwrap(), 1.0);
        assert_float_eq_f64(eval_str("--1").unwrap(), 1.0);
        assert_float_eq_f64(eval_str("----1").unwrap(), 1.0);
        assert_float_eq_f64(eval_str("---1").unwrap(), -1.0);
        assert_float_eq_f64(eval_str("3-(4-2/3+(1-2*2))").unwrap(), 2.666666666666666);
        assert_float_eq_f64(
            eval_str("log(log(2))*tan(2)+exp(1.5)").unwrap(),
            5.2825344122094045,
        );
        assert_float_eq_f64(
            eval_str("log(log2(2))*tan(2)+exp(1.5)").unwrap(),
            4.4816890703380645,
        );
        assert_float_eq_f64(eval_str("log2(2)").unwrap(), 1.0);
        assert_float_eq_f64(eval_str("2^log2(2)").unwrap(), 2.0);
        assert_float_eq_f64(eval_str("2^(cos(0)+2)").unwrap(), 8.0);
        assert_float_eq_f64(eval_str("2^cos(0)+2").unwrap(), 4.0);
    }

    #[test]
    fn test_error_handling() {
        assert!(eval_str::<f64>("").is_err());
        assert!(eval_str::<f64>("5+5-(").is_err());
        assert!(eval_str::<f64>(")2*(5+5)*3-2)*2").is_err());
        assert!(eval_str::<f64>("2*(5+5))").is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_public_interface() {
        let s = "{x}^(3.0-{y})";
        let flatex = FlatEx::<f64>::from_str(s).unwrap();
        let serialized = serde_json::to_string(&flatex).unwrap();
        let deserialized = serde_json::from_str::<FlatEx<f64>>(serialized.as_str()).unwrap();
        assert_eq!(s, format!("{}", deserialized));
    }
}
