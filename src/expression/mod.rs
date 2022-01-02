use std::{fmt::Debug, str::FromStr};

use crate::{data_type::DataType, parser, ExResult};
use num::Float;
use regex::Regex;

pub mod deep;
mod deep_details;
pub mod flat;
mod flat_details;
mod partial_derivatives;
#[cfg(feature = "serde")]
mod serde;

/// Expressions implementing this trait can be evaluated for specific variable values,
/// differentiated partially, and unparsed, i.e., transformed into a string representation.  
pub trait Express<'a, T> {
    /// Parses a string into an expression that can be evaluated.
    ///
    /// # Arguments
    ///
    /// * `text` - string to be parsed into an expression
    ///
    /// # Errors
    ///
    /// An error is returned if `text` cannot be parsed.
    ///
    fn from_str(text: &'a str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: FromStr,
        Self: Sized;

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

    /// This method computes a new instance that is a partial derivative of
    /// `self` with default operators.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    ///
    /// let expr = FlatEx::<f64>::from_str("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.clone().partial(0)?;
    /// let dexpr_dy = expr.partial(1)?;
    ///
    /// assert!((dexpr_dx.eval(&[9e5, 2.0])? - (5.0 as f64).sin()).abs() < 1e-12);
    /// //             |    
    /// //           The partial derivative dexpr_dx does depend on x. Still, it
    /// //           expects the same number of parameters as the corresponding
    /// //           antiderivative. Hence, you can pass any number for x.  
    ///
    /// assert!((dexpr_dy.eval(&[2.5, 2.0])? - 10.0 * (5.0 as f64).cos()).abs() < 1e-12);
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    /// # Arguments
    ///
    /// * `var_idx` - variable with respect to which the partial derivative is computed
    ///
    /// # Errors
    ///
    /// * If `self` has been [`reduce_memory`](Express::reduce_memory)ed, we cannot compute the partial derivative and return an [`ExError`](super::result::ExError).
    /// * If you use custom operators this might not work as expected. It could return an [`ExError`](super::result::ExError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial(self, var_idx: usize) -> ExResult<Self>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug;

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
    fn unparse(&self) -> ExResult<String>;

    /// This function frees some memory. After calling [`partial`](Express::partial) memory might
    /// be re-allocated.
    fn reduce_memory(&mut self);

    /// Returns the number of variables of the expression
    fn n_vars(&self) -> usize;
}

/// Implement this trait to create a matcher for custom literals of operands.
pub trait MakeLiteralMatcher {
    /// Returns a function that returns `Some(matching_str)` in case a string was 
    /// match and None otherwise. 
    fn make() -> fn(&str) -> Option<&str>;
}

/// Default factory to match numeric literals.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct NumberMatcherFactory;
impl MakeLiteralMatcher for NumberMatcherFactory {
    fn make() -> fn(&str) -> Option<&str> {
        parser::is_numeric_text
    }
}

/// Utility function that converts a regex match into matcher of the form expected by the trait
/// [`MakeLiteralMatcher`](MakeLiteralMatcher).
pub fn matches_regex<'a>(re: &Regex, text: &'a str) -> Option<&'a str> {
    let maybe_num = re.find(text);
    match maybe_num {
        Some(m) => Some(m.as_str()),
        None => None,
    }
}
#[macro_export]
macro_rules! literal_matcher_factory {
    ($factory_name:ident, $str:expr) => {
        #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
        pub struct $factory_name;
        impl MakeLiteralMatcher for $factory_name {
            fn make() -> fn(&str) -> Option<&str> {
                fn matcher(text: &str) -> Option<&str> {
                    lazy_static::lazy_static! {
                        static ref RE_VAR_NAME_EXACT: regex::Regex = regex::Regex::new($str).unwrap();
                    }
                    exmex::matches_regex(&RE_VAR_NAME_EXACT, text)
                }
                matcher
            }
        }
    }
}
