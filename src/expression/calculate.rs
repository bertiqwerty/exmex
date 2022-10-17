use std::{fmt::Debug, str::FromStr};

use crate::{data_type::DataType, DeepEx, ExResult, Express};


/// Calculation with expression such as application of operators or substitution
pub trait Calculate<'a, T>: Express<'a, T>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    Self: Sized,
{
    /// Applies a unary operator.
    /// 
    /// # Arguments
    ///
    /// * `repr` - representation of the binary operator
    ///
    /// # Errors
    /// 
    /// * The passed string can be found in the list of operator representations.
    /// 
    fn operate_unary(self, repr: &'a str) -> ExResult<Self> {
        let deepex = self.to_deepex()?;
        let operated = deepex.operate_unary(repr)?;
        Self::from_deepex(operated)
    }
    /// Applies a binary operator.
    /// 
    /// # Arguments
    /// 
    /// * `other` - other expression to be used in the in the binary operation 
    /// * `repr` - representation of the binary operator
    ///
    /// # Errors
    /// 
    /// * The passed string can be found in the list of operator representations.
    /// 
    fn operate_binary(self, other: Self, repr: &'a str) -> ExResult<Self> {
        let deepex = self.to_deepex()?;
        let operated = deepex.operate_bin(other.to_deepex()?, repr)?;
        Self::from_deepex(operated)
    }

    /// Substitutes a variable with another expression.
    /// 
    /// # Arguments
    /// 
    /// * `sub` - function that assigns to each variable name optionally a new expression
    /// 
    /// # Example
    /// 
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::{DeepEx, prelude::*};
    /// use std::f64;
    /// let expr = DeepEx::<f64>::parse("-z/x + 2^7 + E")?;
    /// let mut sub = |var: &str| match var {
    ///     "z" => Some(DeepEx::<f64>::parse("2*y").unwrap()),
    ///     _ => None,
    /// };
    /// let substituted = expr.subs(&mut sub)?;
    /// assert_eq!(substituted.var_names(), ["x", "y"]);
    /// let reference = 2.0f64.powi(7) + f64::consts::E + 1.0;
    /// assert_eq!(substituted.eval(&[-4.0, 2.0])?, reference);
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    fn subs<F>(self, sub: &mut F) -> ExResult<Self> 
    where
        F: FnMut(&str) -> Option<Self>
    {
        let mut sub_deepex = |var: &str| sub(var).and_then(|e|e.to_deepex().ok());
        let substituted = self.to_deepex()?.subs(&mut sub_deepex);
        Self::from_deepex(substituted)
    }
    
    /// Create an expression that contains exactly one number.
    /// 
    /// # Arguments
    /// 
    /// * `x` - number the expression will represent
    /// 
    fn from_num(x: T) -> Self {
        Self::from_deepex(DeepEx::from_num(x))
            .expect("we expect expressions to be constructable from a number")
    }
}

/// For floats we provide ways to conviniently create expressions that represent a `1` or a `0`.
pub trait CalculateFloat<'a, T>: Calculate<'a, T>
where
    T: DataType + num::Float,
    <T as FromStr>::Err: Debug,
    Self: Sized,
{
    fn zero() -> Self {
        Self::from_deepex(DeepEx::zero())
            .expect("we expect expressions to be constructable from a 0")
    }
    fn one() -> Self {
        Self::from_deepex(DeepEx::one())
            .expect("we expect expressions to be constructable from a 1")
    }
}
