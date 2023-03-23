use std::{fmt::Debug, str::FromStr};

pub trait DataType: Clone + FromStr + Debug + Default {}
impl<T: Clone + FromStr + Debug + Default> DataType for T {}
