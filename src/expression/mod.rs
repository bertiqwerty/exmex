use std::fmt::Debug;

use crate::{parser, ExResult, MakeOperators};

pub mod flat;
#[cfg(feature = "serde")]
mod serde;

/// Expressions implementing this trait can be parsed from stings,
/// evaluated for specific variable values, and unparsed, i.e.,
/// transformed into a string representation.  
pub trait Express<T>
where
    T: Clone,
{
    type OperatorFactory: MakeOperators<T>;
    type LiteralMatcher: MatchLiteral;

    /// Evaluates an expression with the given variable values and returns the computed
    /// result.
    ///
    /// # Arguments
    ///
    /// * `vars` - Values of the variables of the expression; the n-th value corresponds to
    ///            the n-th variable in alphabetical order.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///            If an expression has been created by partial derivation, the variables always
    ///            coincide with those of the antiderivatives even in cases where variables are
    ///            irrelevant such as `(x)'=1`.
    ///
    /// # Errors
    ///
    /// If the number of variables in the parsed expression are different from the length of
    /// the variable slice, we return an [`ExError`](super::result::ExError).
    ///
    fn eval(&self, vars: &[T]) -> ExResult<T>;

    /// Evaluates an expression with the given variable values and returns the computed
    /// result. If more variables are passed than necessary the unnecessary ones are ignored.
    ///
    /// # Arguments
    ///
    /// * `vars` - Values of the variables of the expression; the n-th value corresponds to
    ///            the n-th variable in alphabetical order.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///            If an expression has been created by partial derivation, the variables always
    ///            coincide with those of the antiderivatives even in cases where variables are
    ///            irrelevant such as `(x)'=1`. 
    ///
    /// # Errors
    ///
    /// If the number of variables in the parsed expression is larger than the length of
    /// the variable slice, we return an [`ExError`](super::result::ExError).
    ///
    fn eval_relaxed(&self, vars: &[T]) -> ExResult<T>;

    /// Creates an expression string that corresponds to the `FlatEx` instance.
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    /// let flatex = FlatEx::<f64>::from_str("--sin ( z) +  {another var} + 1 + 2")?;
    /// assert_eq!(format!("{}", flatex), "--sin ( z) +  {another var} + 1 + 2");
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    ///
    fn unparse(&self) -> &str;

    /// Returns the variables of the expression
    fn var_names(&self) -> &[String];
}

/// Implement this trait to create a matcher for custom literals of operands.
pub trait MatchLiteral {
    /// This method is expected to return `Some(matching_str)` in case of a match of
    /// a literal at the beginning of the input and `None` otherwise.
    fn is_literal(text: &str) -> Option<&str>;
}

/// Default factory to match numeric literals.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct NumberMatcher;
impl MatchLiteral for NumberMatcher {
    fn is_literal(text: &str) -> Option<&str> {
        parser::is_numeric_text(text)
    }
}

/// Helper to implement a struct called `$matcher_name` that implements
/// [`MatchLiteral`](MatchLiteral) and matches the regex pattern `$regex_pattern`.
///
/// For instance, to match only boolean literals one can create a struct with name
/// `BooleanMatcher` via
/// ```rust
/// use exmex::{literal_matcher_from_pattern, MatchLiteral};
/// literal_matcher_from_pattern!(BooleanMatcher, "^(true|false)");
/// ```
#[macro_export]
macro_rules! literal_matcher_from_pattern {
    ($matcher_name:ident, $regex_pattern:expr) => {
        /// Literal matcher type that was created with the macro
        /// [`literal_matcher_from_pattern`](literal_matcher_from_pattern).
        #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
        pub struct $matcher_name;
        impl MatchLiteral for $matcher_name {
            fn is_literal(text: &str) -> Option<&str> {
                lazy_static::lazy_static! {
                    static ref RE_VAR_NAME_EXACT: regex::Regex = regex::Regex::new($regex_pattern).unwrap();
                }
                RE_VAR_NAME_EXACT.find(text).map(|m|m.as_str())
            }
        }
    };
}
