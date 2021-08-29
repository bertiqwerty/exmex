use num::Float;

use crate::ExParseError;

pub mod deep;
mod deep_details;
pub mod flat;
mod flat_details;
mod partial_derivatives;
#[cfg(feature = "serde_support")]
mod serde;

pub trait Expression<T: Copy> {
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
    /// the variable slice, we return an [`ExParseError`](ExParseError).
    ///
    fn eval(&self, vars: &[T]) -> Result<T, ExParseError>;

    /// This method computes an `Expression` instance that is a partial derivative of
    /// `self` with default operators as shown in the following example for a [`FlatEx`](super::flat::FlatEx).
    ///
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::Expression;
    ///
    /// let expr = exmex::parse_with_default_ops::<f64>("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.clone().partial(0)?;
    /// let dexpr_dy = expr.partial(1)?;
    ///
    /// assert!((dexpr_dx.eval(&[9e5, 2.0])? - (5.0 as f64).sin()).abs() < 1e-12);
    /// //                   |    
    /// //             This partial derivative dexpr_dx does depend on x. Still, it expects
    /// //             the same number of parameters as the corresponding
    /// //             antiderivative. Hence, you can pass any number for x.  
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
    /// * If `self` has been `clear_deepex`ed, we cannot compute the partial derivative and return an [`ExParseError`](ExParseError).
    /// * If you use none-default operators this might not work as expected. It could return an [`ExParseError`](ExParseError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial(self, var_idx: usize) -> Result<Self, ExParseError>
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
    /// use exmex;
    /// let flatex = exmex::parse_with_default_ops::<f64>("--sin ( z) +  {another var} + 1 + 2")?;
    /// assert_eq!(format!("{}", flatex), "-(-(sin({z})))+{another var}+3.0");
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    ///
    fn unparse(&self) -> Result<String, ExParseError>;

    /// This function frees some memory. After calling this, the methods [`partial`](Expression::partial) and
    /// [`unparse`](Expression::unparse) as well as the implementation of the
    /// [`Display`](std::fmt::Display) trait might stop working.
    fn reduce_memory(&mut self);
}
