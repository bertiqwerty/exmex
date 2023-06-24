use std::{fmt::Debug, str::FromStr};

pub trait NeutralElts: PartialEq {
    fn one() -> Self;
    fn zero() -> Self;
}

impl<T> NeutralElts for T
where
    T: From<u8> + PartialEq,
{
    fn one() -> Self {
        T::from(1)
    }
    fn zero() -> Self {
        T::from(0)
    }
}

/// Gathers `Clone`, `FromStr`, `Debug`, and `Default` in one trait.
/// Every type that is used as value needs to implement at least this.
pub trait DataType: Clone + FromStr + Debug + Default {}
impl<T: Clone + FromStr + Debug + Default> DataType for T {}
