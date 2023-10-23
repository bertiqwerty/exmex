use std::{fmt::Debug, str::FromStr};

/// The neutral elements of addition and multiplication are zero and one, respectively.
/// An implementation is provided for all types that implement `From<u8> + PartialEq`.
pub trait NeutralElts: PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
}

impl<T> NeutralElts for T
where
    T: From<u8> + PartialEq,
{
    fn zero() -> Self {
        T::from(0)
    }
    fn one() -> Self {
        T::from(1)
    }
}

/// Gathers `Clone`, `FromStr`, `Debug`, and `Default` in one trait.
/// Every type that is used as value needs to implement at least this.
pub trait DataType: Clone + FromStr + Debug + Default {}
impl<T: Clone + FromStr + Debug + Default> DataType for T {}

/// [`DataType`]s of expressions that are differentiable need to implement
/// additionally `From<f32>` and [`NeutralElts`]. They are gathered here.
#[cfg(feature = "partial")]
pub trait DiffDataType: DataType + From<f32> + NeutralElts {}
#[cfg(feature = "partial")]
impl<T: DataType + From<f32> + NeutralElts> DiffDataType for T {}
