use std::{fmt::Debug, marker::PhantomData, str::FromStr};

use num::{Float, PrimInt};
use smallvec::SmallVec;

use crate::{data_type::DataType, BinOp, ExError, ExResult, MakeOperators, Operator};

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
                Val::Error(e) => Err(e),
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
                Val::Error(e) => Err(e),
                _ => Err(ExError {
                    msg: format!("{:?} does not contain Array", self),
                }),
            }
        }
    };
}

#[macro_export]
macro_rules! from_type {
    ($name:ident, $scalar_variant:ident, $T:ty) => {
        fn $name(x: $T) -> Val<I, F> {
            Val::<I, F>::Scalar(Scalar::$scalar_variant(x))
        }
    };
}

#[derive(Clone, Debug)]
pub enum Scalar<I: DataType + PrimInt, F: DataType + Float> {
    Int(I),
    Float(F),
    Bool(bool),
}
impl<I, F> Scalar<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
{
    to_type!(to_float, F, Float);
    to_type!(to_int, I, Int);
    to_type!(to_bool, bool, Bool);
}
#[derive(Clone, Debug)]
pub enum Val<I: DataType + PrimInt, F: DataType + Float> {
    Scalar(Scalar<I, F>),
    Array(Array<I, F>),
    // since the trait `Try` is experimental, we keep track of an error in an additional arm
    Error(ExError),  
}

impl<I, F> Val<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
{
    to_scalar_type!(to_int, I);
    to_scalar_type!(to_float, F);
    to_scalar_type!(to_bool, bool);
    to_array_type!(to_int_array, to_int, I);
    to_array_type!(to_float_array, to_float, F);
    to_array_type!(to_bool_array, to_bool, bool);

    from_type!(from_float, Float, F);
    from_type!(from_int, Int, I);
    from_type!(from_bool, Bool, bool);
}

fn map_parse_err<E: Debug>(e: E) -> ExError {
    ExError {
        msg: format!("{:?}", e),
    }
}

fn parse_scalar<I, F>(s: &str) -> ExResult<Scalar<I, F>>
where
    I: DataType + PrimInt,
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
    I: DataType + PrimInt,
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

/// Factory of default operators for floating point values.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct ValOpsFactory<I: PrimInt, F: Float> {
    dummy_i: PhantomData<I>,
    dummy_f: PhantomData<F>,
}

fn unpack<I, F>(v: ExResult<Val<I, F>>) -> Val<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
{
    match v {
        Err(e) => Val::<I, F>::Error(e),
        Ok(v) => v,
    }
}

fn pow_scalar<I, F>(a: Scalar<I, F>, b: Scalar<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
{
    match a {
        Scalar::Float(xa) => match b {
            Scalar::Float(xb) => Val::<I, F>::from_float(xa.powf(xb)),
            Scalar::Int(nb) => {
                let powered = xa.powi(nb.to_i32().unwrap());
                unpack(Ok(Val::<I, F>::from_float(powered)))
            }
            Scalar::Bool(_) => Val::Error(ExError::from_str("cannot use bool as exponent")),
        },
        Scalar::Int(na) => match b {
            Scalar::Float(_) => Val::Error(ExError::from_str("cannot use float as exponent")),
            Scalar::Int(nb) => {
                let powered = na.pow(nb.to_u32().unwrap());

                unpack(Ok(Val::<I, F>::from_int(powered)))
            }
            Scalar::Bool(_) => Val::Error(ExError::from_str("cannot use bool as exponent")),
        },
        Scalar::Bool(_) => Val::Error(ExError::from_str("cannot use bool as base")),
    }
}

fn pow<I, F>(base: Val<I, F>, exponent: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
{
    match base {
        Val::Scalar(x) => match exponent {
            Val::Scalar(y) => pow_scalar(x, y),
            Val::Array(_) => Val::Error(ExError::from_str("cannot use array as exponent")),
            Val::Error(e) => Val::Error(e),
        },
        Val::Array(aa) => match exponent {
            Val::Scalar(y) => {
                let powered = aa
                    .iter()
                    .map(|xi| match pow_scalar(xi.clone(), y.clone()) {
                        Val::Scalar(s) => Ok(s),
                        Val::Array(_) => Err(ExError::from_str("cannot build array of arrays")),
                        Val::Error(e) => Err(ExError::from_str(e.msg.as_str())),
                    })
                    .collect::<ExResult<Array<I, F>>>();
                match powered {
                    Err(e) => Val::Error(e),
                    Ok(a) => Val::Array(a),
                }
            }
            Val::Array(_) => Val::Error(ExError::from_str("cannot use array as exponent")),
            Val::Error(e) => Val::Error(e),
        },
        Val::Error(e) => Val::Error(e),
    }
}

impl<I, F> MakeOperators<Val<I, F>> for ValOpsFactory<I, F>
where
    I: DataType + PrimInt,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    /// Returns the default operators.
    fn make<'a>() -> Vec<Operator<'a, Val<I, F>>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: |a, b| pow(a, b),
                    prio: 4,
                    is_commutative: false,
                },
            )
        ]
    }
}

#[cfg(test)]
mod tests {

    use crate::{
        value::{Scalar, Val},
        ExError, ExResult, Express, FlatEx,
    };
    use smallvec::smallvec;

    use super::ValOpsFactory;
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
            Val::<i32, f64>::Array(smallvec![Scalar::Int(1), Scalar::Int(1), Scalar::Int(1)])
                .to_int()
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

    #[test]
    fn test() -> ExResult<()> {
        let expr = FlatEx::<Val<i32, f64>, ValOpsFactory<i32, f64>>::from_str("2^4").unwrap();
        assert_eq!(16, expr.eval(&[])?.to_int()?);
        Ok(())
    }
}
