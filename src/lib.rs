#![doc(html_root_url = "https://docs.rs/exmex/0.20.0")]
//! Exmex is an extendable mathematical expression parser and evaluator. Ease of use, flexibility, and efficient evaluations are its main design goals.
//! Exmex can parse mathematical expressions possibly containing variables and operators. On the one hand, it comes with a list of default operators
//! for floating point values. For differentiable default operators, Exmex can compute partial derivatives. On the other hand, users can define their
//! own operators and work with different data types such as float, integer, bool, or other types that implement `Clone`, `FromStr`, `Debug`, and Default.
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
//! For floats, we have a list of predefined operators containing
//! `^`, `*`, `/`, `+`, `-`, `sin`, `cos`, `tan`, `exp`, `log10`, `ln`, and `log2`. Further, the constants π, τ,
//! and Euler's number are refered to via `π`/`PI`, `τ/TAU`, and `E`, respectively. The full list is
//! defined in [`FloatOpsFactory`]. Library users can also create their
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
//! The value returned by [`parse`] is an instance of the struct [`FlatEx`(
//! that implements the [`Express`] trait. Moreover, [`FlatEx`],
//! [`Express`], and [`Calculate`] are the items made accessible by the
//! wildcard import from [`prelude`] if the feature `partial` is not used.
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
//! [`FlatEx`].
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
//! [`Operator`]. Constants are defined in terms of constant operators. More precisely,
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
//! All binary operators can be used either like `a op b` or like `op(a, b)`. Thereby, the latter will be interpreted as `((a) op (b))`. For instance
//! `atan2(y * 2, 1 / x) * 2` and `((y * 2) atan2 (1 / x)) * 2` are equivalent. We do not support `n`-ary operators like `f(a, b, c)` for `n = 3`.
//!
//! Binary, unary, and constant operators can be created with the functions [`make_bin`](Operator::make_bin), [`make_unary`](Operator::make_unary),
//! and [`make_constant`](Operator::make_constant), respectively.
//! Operators need to be created by factories to make serialization via [`serde`](https://serde.rs/) possible as
//! shown in the following.
//!
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
//! To extend an existing list of operators, the macro [`ops_factory`] is not
//! sufficient. In this case one has to create a factory struct and implement the
//! [`MakeOperators`] trait with a little boilerplate code.
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
//! You can use any type that implements [`Clone`],
//! [`FromStr`], and [`Debug`]. In case the representation of your data type's literals
//! in the string does not match the number regex `r"^(\.?[0-9]+(\.[0-9]+)?)"`, you have to create a suitable matcher
//! type that implements [`MatchLiteral`]. Given a suitable regex pattern, you can utilize the macro
//! [`literal_matcher_from_pattern`].
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
//! Like partial derivatives, calculations need the nested expression type [`DeepEx`](`DeepEx`) that is
//! slower to evaluate than the flattened expression type [`FlatEx`](`FlatEx`). It is possible to calculate
//! with flat expressions of type [`FlatEx`](`FlatEx`). However, transformations to the
//! nested expression [`DeepEx`](`DeepEx`) happen in the background.
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use exmex::prelude::*;
//! let expr_1 = FlatEx::<f64>::parse("x")?;
//! let expr_2px = FlatEx::<f64>::parse("2 + x")?;
//! let expr_2p2x = expr_1.operate_binary(expr_2px, "+")?;
//! assert!(expr_2p2x.eval(&[-1.5])? < 1e-12);
//! #
//! #     Ok(())
//! # }
//!```
//!
//! To save transformations, we can start by parsing a deep expression to do multiple calculations
//! and flatten eventually.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::{DeepEx, prelude::*};
//! let deep_cos_x = DeepEx::<f64>::parse("cos(x)")?;
//! let deep_identity = deep_cos_x.operate_unary("acos")?;
//! let one = DeepEx::one();
//! let deep_identity = deep_identity.operate_binary(one, "*")?;
//! let flat_identity = FlatEx::from_deepex(deep_identity)?;
//! assert!((flat_identity.eval(&[3.0])? - 3.0).abs() < 1e-12);
//! #
//! # Ok(())
//! # }
//! ```
//! Alternatively, it is possible to transform a flat expression to a nested expression
//! with [`FlatEx::to_deepex`](`FlatEx::to_deepex`). Moreover, we have implemented the default
//! operators as wrappers around [`Calculate::operate_unary`] and
//! [`Calculate::operate_binary`], see the following re-write of the snippet
//! above.
//!
//! ```rust
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! #
//! use exmex::{DeepEx, prelude::*};
//! let deep_cos_x = DeepEx::<f64>::parse("cos(x)")?;
//! let deep_identity = deep_cos_x.acos()?;
//! let one = DeepEx::one();
//! let deep_identity = (deep_identity * one)?;
//! let flat_identity = FlatEx::from_deepex(deep_identity)?;
//! assert!((flat_identity.eval(&[3.0])? - 3.0).abs() < 1e-12);
//! #
//! # Ok(())
//! # }
//! ```

use std::{fmt::Debug, str::FromStr};

mod definitions;
mod expression;
#[macro_use]
mod operators;
mod data_type;
mod parser;
mod result;
#[doc(hidden)]
pub mod statements;
mod util;

#[cfg(feature = "partial")]
pub use data_type::DiffDataType;

#[doc(hidden)]
#[cfg(feature = "value")]
pub use statements::{line_2_statement_val, StatementsVal};
#[doc(hidden)]
pub use statements::{Statement, Statements};
pub use {
    data_type::{DataType, NeutralElts},
    expression::{
        calculate::Calculate, deep::DeepEx, flat::FlatEx, Express, MatchLiteral, NumberMatcher,
    },
    operators::{BinOp, FloatOpsFactory, MakeOperators, Operator},
    result::{ExError, ExResult},
};

// Re-exported since used in macro literal_matcher_from_pattern
pub use {lazy_static, regex};

#[cfg(feature = "value")]
mod value;
#[cfg(feature = "partial")]
pub use expression::partial::{Differentiate, MissingOpMode};
use num::Float;
#[cfg(feature = "value")]
pub use value::{parse_val, ArrayType, FlatExVal, Val, ValMatcher, ValOpsFactory};

/// Exmex' prelude can be imported via `use exmex::prelude::*;`.
///
/// The prelude contains
/// * expression trait [`Express`],
/// * its implementation [`FlatEx`],
/// * and the partial differentiation of [`FlatEx`], if the feature `partial` is active.
///
pub mod prelude {
    pub use crate::expression::{calculate::Calculate, flat::FlatEx, Express};
    #[cfg(feature = "partial")]
    pub use crate::Differentiate;
    pub use std::str::FromStr;
}

/// Parses a string, evaluates the expression, and returns the resulting number.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`] is returned.
///
pub fn eval_str<T: DataType>(text: &str) -> ExResult<T>
where
    T: DataType + Float,
    <T as FromStr>::Err: Debug,
{
    let flatex = FlatEx::<T>::parse_wo_compile(text)?;
    if !flatex.var_names().is_empty() {
        return Err(exerr!("input string contains variables, '{}' ", text));
    }
    flatex.eval(&[])
}

/// Parses a string and returns the expression with default operators that can be evaluated.
///
/// # Errrors
///
/// In case the parsing went wrong, e.g., due to an invalid input string, an
/// [`ExError`] is returned.
///
pub fn parse<T>(text: &str) -> ExResult<FlatEx<T>>
where
    T: DataType + Float,
    <T as FromStr>::Err: Debug,
{
    FlatEx::<T>::parse(text)
}
