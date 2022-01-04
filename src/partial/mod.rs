use std::{fmt::Debug, str::FromStr};

use num::Float;

use crate::{data_type::DataType, ExResult, Express};

pub mod deep;
mod deep_details;
pub mod partial_derivatives;

/// *`feature = "partial"`* - Trait for partial differentiation.  
pub trait Differentiate<T, Ex> {
    /// *`feature = "partial"`* - This method computes a new expression 
    /// that is expected to be a partial derivative of`self` with default operators.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    ///
    /// let mut expr = FlatEx::<f64>::from_str("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.partial(0)?;
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
    /// * If you use custom operators this might not work as expected. It could return an [`ExError`](crate::ExError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial(&self, var_idx: usize) -> ExResult<Ex>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug,
        Ex: Express<T>;
}
