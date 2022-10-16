#![doc(html_root_url = "https://docs.rs/exmex/0.16.1")]
//! Exmex is an extendable mathematical expression parser and evaluator. Ease of use, flexibility, and efficient evaluations are its main design goals.
//! Exmex can parse mathematical expressions possibly containing variables and operators. On the one hand, it comes with a list of default operators
//! for floating point values. For differentiable default operators, Exmex can compute partial derivatives. On the other hand, users can define their
//! own operators and work with different data types such as float, integer, bool, or other types that implement `Clone`, `FromStr`, and `Debug`.
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
//! `^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log10`, `ln`, and `log2`. Further, the constants π, τ,
//! and Euler's number are refered to via `π`/`PI`, `τ/TAU`, and `E`, respectively. The full list is
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
//! let to_be_parsed = "α * ln(z) + 2* (-z^2 + sin(4*y))";
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
//! let to_be_parsed = "ln({👍+👎})";  // {👍+👎} is the name of one variable 😕.
//! let expr = exmex::parse::<f64>(to_be_parsed)?;
//! assert!((expr.eval(&[x+y])? - 2.2f64.ln()).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! The value returned by [`parse`](parse) is an instance of the struct [`FlatEx`](FlatEx)
//! that implements the [`Express`](Express) trait. Moreover, [`FlatEx`](FlatEx) and
//! [`Express`](Express) are the only items made accessible by the wildcard import from
//! [`prelude`](prelude) if the feature `partial` is not used.
//!
//! ## Features
//! Exmex comes with three features that can be activated in the `Cargo.toml` via
//! ```text
//! [dependencies]
//! exmex = { ..., features = ["partial", "serde", "value"] }
//! ```
//!
//! `partial` allows the computation of partal derivatives, `serde` enables serialization and
//! deserialization, and `value` makes a more general value type accessible.
//!
//! ### Partial Derivatives
//!
//! Expressions with floating point data types can be transformed into their
//! partial derivatives again represented by expressions after activating the feature `partial`.
//! See the [readme](https://github.com/bertiqwerty/exmex#partial-differentiation) for examples.
//!
//! ### Serialization and Deserialization
//!
//! To use [`serde`](https://serde.rs/) you can activate the feature `serde`.
//! The implementation un-parses and re-parses the whole expression.
//! [`Deserialize`](https://docs.serde.rs/serde/de/trait.Deserialize.html) and
//! [`Serialize`](https://docs.serde.rs/serde/de/trait.Serialize.html) are implemented for
//! [`FlatEx`](FlatEx).
//!
//! ### A more General Value Type
//!
//! To use different data types within an expression, one can activate the feature `value` and
//! use the more general type `Val`. The additional flexibility comes with higher parsing
//! and evaluation run times, see the [benchmarks](https://github.com/bertiqwerty/exmex#benchmarks-v0130).
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
//! An operator's representation can be accessed via the method
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
//! let expr = FlatEx::<_, IntegerOpsFactory>::parse(to_be_parsed)?;
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
//! #[derive(Clone, Debug)]
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
//! let expr = FlatEx::<_, ExtendedOpsFactory>::parse(to_be_parsed)?;
//! assert!((expr.eval(&[3.0])? - 2.0/3.0).abs() < 1e-12);
//! #
//! #     Ok(())
//! # }
//! ```
//! ### Custom Data Types of Numbers
//!
//! You can use any type that implements [`Clone`](Clone),
//! [`FromStr`](std::str::FromStr), and [`Debug`](std::fmt::Debug). In case the representation of your data type's literals
//! in the string does not match the number regex `r"^(\.?[0-9]+(\.[0-9]+)?)"`, you have to create a suitable matcher
//! type that implements [`MatchLiteral`](MatchLiteral). Given a suitable regex pattern, you can utilize the macro
//! [`literal_matcher_from_pattern`](literal_matcher_from_pattern).
//! Here is an example for `bool`.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! use exmex::{
//!     BinOp, MakeOperators, MatchLiteral, Operator,
//!     literal_matcher_from_pattern, ops_factory
//! };
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
//! literal_matcher_from_pattern!(BooleanMatcher, "^(true|false)");
//! let to_be_parsed = "!(true && false) || (!false || (true && false))";
//! type FlatExBool = FlatEx::<bool, BooleanOpsFactory, BooleanMatcher>;
//! let expr = FlatExBool::parse(to_be_parsed)?;
//! assert_eq!(expr.eval(&[])?, true);
//! #
//! #     Ok(())
//! # }
//! ```
//! Two examples of exmex with non-trivial data types are:
//! * Numbers can be operators and operators can operate on operators, see, e.g.,
//! also a blog post on [ninety.de](https://www.ninety.de/log/index.php/en/2021/11/11/parsing-operators-in-rust/).
//! * The value type implemented as part of the feature `value` allows expressions containing integers, floats, and bools.
//! Therewith, Pythonesque expressions of the form `"x if a > b else y"` are possible.
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
//! Expressions can be displayed as string. This
//! [`unparse`](Express::unparse)d string coincides with the original
//! string.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let expr = exmex::parse::<f64>("-sin(z)/cos(mother_of_names) + 2^7 + E")?;
//! assert_eq!(format!("{}", expr), "-sin(z)/cos(mother_of_names) + 2^7 + E");
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! ## Calculating with Expression
//!
//! One cannot calculate with flattened expression of type [`FlatEx`](`FlatEx`) directly. However,
//! one can apply all defined operators to nested expressions of type [`DeepEx`](`DeepEx`).
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::{DeepEx, prelude::*};
//! let deepex_1 = DeepEx::one();
//! let deepex_2px = DeepEx::<f64>::parse("2 + x")?;
//! let deepex_3px = deepex_1.operate_bin_repr(deepex_2px, "+")?;
//! assert_eq!(format!("{}", deepex_3px), "1.0+(2.0+{x})");
//! assert!(deepex_3px.eval(&[-3.0])? < 1e-12);
//! #
//! #     Ok(())
//! # }
//!```
//! The method [`operate_bin_repr`](`DeepEx::operate_bin_repr`) traverses all operators to find the
//! right one. It is possible to do this separately as shown in the following with a unary operator
//! as example.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::{DeepEx, prelude::*};
//! let deepex = DeepEx::<f64>::parse("x")?;
//! let ops = deepex.make_ops();
//! let sin_op = exmex::find_unary_op("sin", &ops)?;
//! let deep_sin_x = deepex.operate_unary(sin_op);
//! assert!((deep_sin_x.eval(&[1.0])? - (1.0f64).sin()).abs() < 1e-12);
//! 
//! // for faster repeated evaluation, we can flatten the expression
//! let flat_sin_x = FlatEx::from_deepex(deep_sin_x.clone())?;
//! for i in 0..1000 {
//!     assert!((deep_sin_x.eval(&[i as f64])? - flat_sin_x.eval(&[i as f64])?).abs() < 1e-12);
//! }
//! 
//! #
//! #     Ok(())
//! # }
//! ```
//!
//! If we start by parsing a flat expression, we can deepen the expression to do calculations.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::prelude::*;
//! let flat_cos_x = FlatEx::<f64>::parse("cos(x)")?;
//! let deep_cos_x = flat_cos_x.to_deepex()?;
//! let deep_identity = deep_cos_x.operate_unary_repr("acos")?;
//! assert!((deep_identity.eval(&[3.0])? - 3.0).abs() < 1e-12);
//! #
//! # Ok(())
//! # }
//! ```

use std::{fmt::Debug, str::FromStr};

use data_type::DataType;
use num::Float;
mod definitions;
mod expression;
#[macro_use]
mod operators;
mod data_type;
mod parser;
mod result;
mod util;

pub use {
    expression::{
        deep::find_bin_op, deep::find_unary_op, deep::DeepEx, flat::FlatEx, Express, MatchLiteral,
        NumberMatcher,
    },
    operators::{BinOp, FloatOpsFactory, MakeOperators, Operator},
    result::{ExError, ExResult},
};

// Re-exported since used in macro literal_matcher_from_pattern
pub use {lazy_static, regex};

#[cfg(feature = "value")]
mod value;
#[cfg(feature = "value")]
pub use value::{parse_val, FlatExVal, Val, ValMatcher, ValOpsFactory};
#[cfg(feature = "partial")]
pub use expression::partial::Differentiate;

/// Exmex' prelude can be imported via `use exmex::prelude::*;`.
///
/// The prelude contains
/// * expression trait [`Express`](Express),
/// * its implementation [`FlatEx`](FlatEx),
/// * and the partial differentiation of [`FlatEx`](FlatEx), if the feature `partial` is active.
///
pub mod prelude {
    pub use crate::expression::{flat::FlatEx, Express};
    #[cfg(feature = "partial")]
    pub use crate::Differentiate;
    pub use std::str::FromStr;
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
    let flatex = FlatEx::<T>::parse_wo_compile(text)?;
    if !flatex.var_names().is_empty() {
        return Err(format_exerr!(
            "input string contains variables, '{}' ",
            text
        ));
    }
    flatex.eval(&[])
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
    FlatEx::<T>::parse(text)
}
