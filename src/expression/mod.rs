use std::{fmt::Debug, str::FromStr};

use num::Float;

use crate::ExResult;

pub mod deep;
mod deep_details;
pub mod flat;
mod flat_details;
mod partial_derivatives;
#[cfg(feature = "serde")]
mod serde;

/// Expressions implementing this trait can be evaluated for specific variable values,
/// differentiated partially, and unparsed, i.e., transformed into a string representation.  
pub trait Express<'a, T: Copy> {
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
        T: Copy + FromStr,
        Self: Sized;

    /// Use custom number literals defined as regex patterns to create an expression that can be evaluated.
    ///
    /// # Arguments
    ///
    /// * `text` - string to be parsed into an expression
    /// * `number_regex_pattern` - regex pattern whose matches are number literals
    ///
    /// # Errors
    ///
    /// An [`ExError`](super::result::ExError) is returned, if
    ///
    /// * the argument `number_regex_pattern` cannot be compiled or
    /// * the text cannot be parsed.
    ///
    fn from_pattern(text: &'a str, number_regex_pattern: &str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
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
        T: Float;

    /// Creates an expression string that corresponds to the `FlatEx` instance. This is
    /// not necessarily the input string. More precisely,
    /// * variables are put between curly braces,
    /// * spaces outside of curly brackets are ignored,
    /// * parentheses can be different from the input, and
    /// * expressions are compiled
    /// as shown in the following example.
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    /// let flatex = FlatEx::<f64>::from_str("--sin ( z) +  {another var} + 1 + 2")?;
    /// assert_eq!(format!("{}", flatex), "-(-(sin({z})))+{another var}+3.0");
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    ///
    fn unparse(&self) -> ExResult<String>;

    /// This function frees some memory. After calling this, the methods [`partial`](Express::partial) and
    /// [`unparse`](Express::unparse) as well as the implementation of the
    /// [`Display`](std::fmt::Display) trait might stop working.
    fn reduce_memory(&mut self);
}
