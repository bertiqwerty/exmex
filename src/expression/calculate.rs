use std::{fmt::Debug, str::FromStr};

use crate::{data_type::DataType, DeepEx, ExResult, Express};

pub trait Calculate<'a, T>: Express<'a, T>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    Self: Sized,
{
    fn operate_unary(self, repr: &'a str) -> ExResult<Self> {
        let deepex = self.to_deepex()?;
        let operated = deepex.operate_unary(repr)?;
        Self::from_deepex(operated)
    }
    fn operate_binary(self, other: Self, repr: &'a str) -> ExResult<Self> {
        let deepex = self.to_deepex()?;
        let operated = deepex.operate_bin(other.to_deepex()?, repr)?;
        Self::from_deepex(operated)
    }
    fn subs<F>(self, sub: &mut F) -> ExResult<Self> 
    where
        F: FnMut(&str) -> Option<Self>
    {
        let mut sub_deepex = |var: &str| sub(var).and_then(|e|e.to_deepex().ok());
        let substituted = self.to_deepex()?.subs(&mut sub_deepex);
        Self::from_deepex(substituted)
    }
    fn from_num(x: T) -> Self {
        Self::from_deepex(DeepEx::from_num(x))
            .expect("we expect expressions to be constructable from a number")
    }
}

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
