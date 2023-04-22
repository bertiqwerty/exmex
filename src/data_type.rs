use std::{fmt::Debug, str::FromStr};

/// Gathers `Clone`, `FromStr`, `Debug`, and `Default` in one trait.
/// Every type that is used as value needs to implement at least this.
pub trait DataType: Clone + FromStr + Debug + Default {}
impl<T: Clone + FromStr + Debug + Default> DataType for T {}
