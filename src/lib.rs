#![doc(html_root_url = "https://docs.rs/exmex/0.11.4")]
//! Exmex is a fast, simple, and **ex**tendable **m**athematical **ex**pression evaluator
//! with the ability to compute [partial derivatives](FlatEx::partial) of expressions.  
//!
//! The following snippet shows how to evaluate a string.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex;
//! let eval_result = exmex::eval_str::<f64>("1.5 * ((cos(2*π) + 23.0) / 2.0)")?;
//! assert!((eval_result - 18.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! For floats, we have a list of predifined operators containing
//! `^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log`, and `log2`. Further, the constants π
//! and Euler's number can be used via `π`/`PI` and `E`, respectively. The full list is
//! defined in [`FloatOpsFactory`](FloatOpsFactory). Library users can also create their
//! own operators and constants as shown below in the section about extendability.
//!
//! ## Variables
//!
//! To define variables we can use strings that are not in the list of operators as shown in the following expression.
//! Additionally, variables should consist only of letters, greek letters, numbers, and underscores. More precisely, they
//! need to fit the regular expression `r"[a-zA-Zα-ωΑ-Ω_]+[a-zA-Zα-ωΑ-Ω_0-9]*"`, if they are not between curly brackets.
//!
//! Variables' values are passed as slices to [`eval`](Express::eval).
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let to_be_parsed = "α * log(z) + 2* (-z^2 + sin(4*y))";
//! let expr = exmex::parse::<f64>(to_be_parsed)?;
//! assert!((expr.eval(&[3.7, 2.5, 1.0])? - 14.992794866624788 as f64).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The `n`-th number in the slice corresponds to the `n`-th variable. Thereby, the
//! alphabetical order of the variables is relevant. More precisely, the order is defined by the way how Rust sorts strings.
//! In the example above we have `y=3.7`, `z=2.5`, and `α=1`. Note that `α` is the Greek letter Alpha.
//! If variables are between curly brackets, they can have arbitrary names, e.g.,
//! `{456/549*(}`, `{x}`, and also `{👍+👎}` are valid variable names as shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let x = 2.1f64;
//! let y = 0.1f64;
//! let to_be_parsed = "log({👍+👎})";  // {👍+👎} is the name of one variable 😕.
//! let expr = exmex::parse::<f64>(to_be_parsed)?;
//! assert!((expr.eval(&[x+y])? - 2.2f64.ln()).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The value returned by [`parse`](parse) is an instance of the struct [`FlatEx`](FlatEx)
//! that implements the [`Express`](Express) trait. Moreover, [`FlatEx`](FlatEx) and
//! [`Express`](Express) are the only items made accessible by the wildcard import from
//! [`prelude`](prelude).
//!
//! ## Partial Derivatives
//!
//! For default operators, expressions can be transformed into their partial derivatives
//! again represented by expressions. To this end, there exists the method [`partial`](Express::partial).
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
//! //              |                          |
//! //              lifetime parameter necessary
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
//! ## Extendability
//!
//! How to use custom operators as well as custom data types of the operands even with
//! non-numeric literals is described in the following sub-sections.
//!
//! ### Custom Operators and Constants
//!
//! Operators are instances of the struct
//! [`Operator`](Operator). Constants are defined in terms of constant operators. More precisely,
//! operators can be
//! * binary such as `*`,
//! * unary such as `sin`,
//! * binary as well as unary such as `-`, or
//! * constant such as `PI`.
//!
//! An operator's representation is defined in the field
//! [`repr`](Operator::repr). A token of the string-to-be-parsed is identified as operator if it matches the operator's
//! representation exactly. For instance, `PI` will be parsed as the constant π while `PI5` will be parsed as a variable with name `PI5`.
//! When an operator's representation is used in a string-to-be-parsed, the following applies:
//! * Binary operators are positioned between their operands, e.g., `4 ^ 5`.
//! * Unary operators are positioned in front of their operands, e.g., `-1` or `sin(4)`. Note that `sin4`
//! is parsed as variable name, but  `sin 4` is equivalent to `sin(4)`.
//! * Constant operators are handled as if they were numbers and are replaced by their numeric values during parsing.
//! They can be used as in `sin(PI)` or `4 + E`. Note that the calling notation of constant operators such as `PI()` is invalid.
//!
//! Binary, unary, and constant operators can be created with the functions [`make_bin`](Operator::make_bin), [`make_unary`](Operator::make_unary),
//! and [`make_constant`](Operator::make_constant), respectively.
//! Operators need to be created by factories to make serialization via [`serde`](https://serde.rs/) possible as
//! shown in the following.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{BinOp, MakeOperators, Operator, ops_factory};
//! ops_factory!(
//!     IntegerOpsFactory,  // name of the factory type
//!     i32,                // data type of the operands
//!     Operator::make_bin(
//!         "%",
//!         BinOp{
//!             apply: |a, b| a % b,
//!             prio: 1,
//!             is_commutative: false,
//!         }
//!     ),
//!     Operator::make_bin(
//!         "/",
//!         BinOp{
//!             apply: |a, b| a / b,
//!             prio: 1,
//!             is_commutative: false,
//!         }
//!     ),
//!     Operator::make_constant("TWO", 2)
//! );
//! let to_be_parsed = "19 % 5 / TWO / a";
//! let expr = FlatEx::<_, IntegerOpsFactory>::from_str(to_be_parsed)?;
//! assert_eq!(expr.eval(&[1])?, 2);
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! To extend an existing list of operators, the macro [`ops_factory`](ops_factory) is not
//! sufficient. In this case one has to create a factory struct and implement the
//! [`MakeOperators`](MakeOperators) trait with a little boilerplate code.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{FloatOpsFactory, MakeOperators, Operator};
//! #[derive(Clone)]
//! struct ExtendedOpsFactory;
//! impl MakeOperators<f32> for ExtendedOpsFactory {
//!     fn make<'a>() -> Vec<Operator<'a, f32>> {
//!         let mut ops = FloatOpsFactory::<f32>::make();
//!         ops.push(
//!             Operator::make_unary("invert", |a| 1.0 / a)
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
//! ### Custom Data Types of Numbers
//!
//! You can use any type that implements [`Clone`](Clone),
//! [`FromStr`](std::str::FromStr), and [`Debug`](std::fmt::Debug). In case the representation of your data type in the
//! string does not match the number regex `r"\.?[0-9]+(\.[0-9]+)?"`, you have to pass a
//! suitable regex and use the function
//! [`from_pattern`](Express::from_pattern) instead of [`parse`](crate::parse) or
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
//!     Operator::make_bin(
//!         "&&",
//!         BinOp{
//!             apply: |a, b| a && b,
//!             prio: 1,
//!             is_commutative: true,
//!         }
//!     ),
//!     Operator::make_bin(
//!         "||",
//!         BinOp{
//!             apply: |a, b| a || b,
//!             prio: 1,
//!             is_commutative: true,
//!         }
//!     ),
//!     Operator::make_unary("!", |a| !a)
//! );
//! let to_be_parsed = "!(true && false) || (!false || (true && false))";
//! let expr = FlatEx::<_, BooleanOpsFactory>::from_pattern(to_be_parsed, "true|false")?;
//! assert_eq!(expr.eval(&[])?, true);
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
//! Instances of [`FlatEx`](FlatEx) and [`OwnedFlatEx`](OwnedFlatEx) can be displayed as string. Note that this
//! [`unparse`](Express::unparse)d string does not necessarily coincide with the original
//! string, since, e.g., curly brackets are added, expressions are compiled, and constants are
//! replaced by their numeric values during parsing.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let expr = exmex::parse::<f64>("-sin(z)/cos(mother_of_names) + 2^7 + E")?;
//! assert_eq!(format!("{}", expr), "-(sin({z}))/cos({mother_of_names})+130.71828182845906");
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

use std::{fmt::Debug, str::FromStr};

use data_type::DataType;
use expression::deep::DeepEx;
use num::Float;
mod definitions;
mod expression;
#[macro_use]
mod operators;
mod data_type;
mod parser;
mod result;
mod util;
mod value;
pub use {
    expression::{
        flat::{FlatEx, OwnedFlatEx},
        Express,
    },
    operators::{BinOp, FloatOpsFactory, MakeOperators, Operator},
    result::{ExError, ExResult},
    value::{parse_val, parse_val_owned, Scalar, Val, FlatExVal, OwnedFlatExVal},
};

/// To use the expression trait [`Express`](Express) and its implementation [`FlatEx`](FlatEx)
/// one can `use exmex::prelude::*;`.
pub mod prelude {
    pub use super::expression::{flat::FlatEx, Express};
}

/// Parses a string, evaluates the expression, and returns the resulting number.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`](ExError) is returned.
///
pub fn eval_str<T: Float + DataType>(text: &str) -> ExResult<T>
where
    <T as FromStr>::Err: Debug,
{
    let deepex = DeepEx::<T>::from_str_float(text)?;
    if deepex.n_vars() > 0 {
        return Err(ExError {
            msg: format!("input string contains variables, '{}' ", text),
        });
    }

    deepex.eval(&[])
}

/// Parses a string and returns the expression that can be evaluated.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`](ExError) is returned.
///
pub fn parse<T: Float + DataType>(text: &str) -> ExResult<FlatEx<T>>
where
    <T as FromStr>::Err: Debug,
{
    FlatEx::<T>::from_str(text)
}
