use std::{fmt::Debug, str::FromStr};

pub trait DataType: Clone + FromStr + Debug {}
impl<T: Clone + FromStr + Debug> DataType for T {}
