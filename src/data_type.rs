use std::{fmt::Debug, str::FromStr};

use num::{Float, Integer};
use smallvec::SmallVec;

use crate::{ExError, ExResult};

pub trait DataType: Clone + FromStr + Debug {}
impl<T: Clone + FromStr + Debug> DataType for T {}
const ARRAY_LEN: usize = 8usize;
pub type Array<I, F> = SmallVec<[Scalar<I, F>; ARRAY_LEN]>;

#[macro_export]
macro_rules! to_type {
    ($name:ident, $T:ty, $variant:ident) => {
        pub fn $name(self) -> ExResult<$T> {
            match self {
                Scalar::$variant(x) => Ok(x),
                _ => Err(ExError {
                    msg: format!(
                        "Scalar {:?} does not contain type {}",
                        self,
                        stringify!($variant)
                    ),
                }),
            }
        }
    };
}

#[macro_export]
macro_rules! to_scalar_type {
    ($name:ident, $T:ty) => {
        pub fn $name(self) -> ExResult<$T> {
            match self {
                Val::Scalar(s) => s.$name(),
                _ => Err(ExError {
                    msg: format!("{:?} does not contain Scalar", self),
                }),
            }
        }
    };
}

#[macro_export]
macro_rules! to_array_type {
    ($name:ident, $scalar_name:ident, $T:ty) => {
        pub fn $name(self) -> ExResult<SmallVec<[$T; ARRAY_LEN]>> {
            match self {
                Val::Array(a) => a
                    .iter()
                    .map(|s| -> ExResult<$T> { s.clone().$scalar_name() })
                    .collect::<ExResult<_>>(),
                _ => Err(ExError {
                    msg: format!("{:?} does not contain Array", self),
                }),
            }
        }
    };
}

#[derive(Clone, Debug)]
pub enum Scalar<I: DataType + Integer, F: DataType + Float> {
    Int(I),
    Float(F),
    Bool(bool),
}
impl<I, F> Scalar<I, F>
where
    I: DataType + Integer,
    F: DataType + Float,
{
    to_type!(to_float, F, Float);
    to_type!(to_int, I, Int);
    to_type!(to_bool, bool, Bool);
}
#[derive(Clone, Debug)]
pub enum Val<I: DataType + Integer, F: DataType + Float> {
    Scalar(Scalar<I, F>),
    Array(Array<I, F>),
}

impl<I, F> Val<I, F>
where
    I: DataType + Integer,
    F: DataType + Float,
{
    to_scalar_type!(to_int, I);
    to_scalar_type!(to_float, F);
    to_scalar_type!(to_bool, bool);
    to_array_type!(to_int_array, to_int, I);
    to_array_type!(to_float_array, to_float, F);
    to_array_type!(to_bool_array, to_bool, bool);
}

fn map_parse_err<E: Debug>(e: E) -> ExError {
    ExError {
        msg: format!("{:?}", e),
    }
}

fn parse_scalar<I, F>(s: &str) -> ExResult<Scalar<I, F>>
where
    I: DataType + Integer,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    let res = Ok(if s.contains(".") {
        Scalar::Float(s.parse::<F>().map_err(map_parse_err)?)
    } else if s == "false" || s == "true" {
        Scalar::Bool(s.parse::<bool>().map_err(map_parse_err)?)
    } else {
        Scalar::Int(s.parse::<I>().map_err(map_parse_err)?)
    });
    match res {
        Result::Ok(_) => res,
        Result::Err(e) => Err(ExError {
            msg: format!("could not parse {}, {:?}", s, e),
        }),
    }
}

impl<I, F> FromStr for Val<I, F>
where
    I: DataType + Integer,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    type Err = ExError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let first = s.chars().nth(0);

        if first == Some('[') {
            let a = &s[1..s.len() - 1];
            Ok(Val::Array(
                a.split(",")
                    .map(|sc_str| -> ExResult<Scalar<I, F>> { parse_scalar(sc_str) })
                    .collect::<ExResult<Array<I, F>>>()?,
            ))
        } else {
            Ok(Val::Scalar(parse_scalar(s)?))
        }
    }
}
#[cfg(test)]
use smallvec::smallvec;
#[test]
fn test_to() -> ExResult<()> {
    assert_eq!(Scalar::<i32, f64>::Float(3.4).to_float()?, 3.4);
    assert_eq!(Scalar::<i32, f64>::Int(123).to_int()?, 123);
    assert!(Scalar::<i32, f64>::Bool(true).to_bool()?);
    assert!(Scalar::<i32, f64>::Bool(false).to_int().is_err());
    assert_eq!(Val::<i32, f64>::Scalar(Scalar::Float(3.4)).to_float()?, 3.4);
    assert_eq!(Val::<i32, f64>::Scalar(Scalar::Int(34)).to_int()?, 34);
    assert!(!Val::<i32, f64>::Scalar(Scalar::Bool(false)).to_bool()?);
    assert!(
        Val::<i32, f64>::Array(smallvec![Scalar::Int(1), Scalar::Int(1), Scalar::Int(1)]).to_int()
            == Err(ExError {
                msg: "Array([Int(1), Int(1), Int(1)]) does not contain Scalar".to_string()
            })
    );
    assert_eq!(
        Val::<i32, f64>::Array(smallvec![Scalar::Int(1), Scalar::Int(2)]).to_int_array(),
        Ok(smallvec![1, 2])
    );
    Ok(())
}
