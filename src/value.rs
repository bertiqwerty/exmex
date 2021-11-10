use std::{fmt::Debug, marker::PhantomData, str::FromStr};

use lazy_static::lazy_static;
use num::{Float, PrimInt, Signed};
use regex::Regex;

use crate::{
    data_type::DataType, format_exerr, BinOp, ExError, ExResult, Express, FlatEx, MakeOperators,
    Operator, OwnedFlatEx,
};

lazy_static! {
    static ref RE_VAR_NAME_EXACT: Regex =
        Regex::new(r"^([0-9]+(\.[0-9]+)?|true|false|\[\s*(\-?.?[0-9]+(\.[0-9]+)?|true|false)(\s*,\s*-?\.?[0-9]+(\.[0-9]+)?|true|false)*\s*\])").unwrap();
}

/// *`feature = "value"`* - Alias for [`FlatEx`](FlatEx) with [`Val`](Val) as data type and [`ValOpsFactory`](ValOpsFactory)
/// as operator factory.
pub type FlatExVal<'a, I, F> = FlatEx<'a, Val<I, F>, ValOpsFactory<I, F>>;
/// *`feature = "value"`* - Alias for [`OwnedFlatEx`](OwnedFlatEx) with [`Val`](Val) as data type and [`ValOpsFactory`](ValOpsFactory)
/// as operator factory.
pub type OwnedFlatExVal<'a, I, F> = OwnedFlatEx<Val<I, F>, ValOpsFactory<I, F>>;

macro_rules! to_type {
    ($name:ident, $T:ty, $variant:ident) => {
        pub fn $name(self) -> ExResult<$T> {
            match self {
                Val::$variant(x) => Ok(x),
                _ => Err(format_exerr!(
                    "value {:?} does not contain type {}",
                    self,
                    stringify!($variant)
                )),
            }
        }
    };
}

macro_rules! from_type {
    ($name:ident, $variant:ident, $T:ty) => {
        pub fn $name(x: $T) -> Val<I, F> {
            Val::<I, F>::$variant(x)
        }
    };
}

/// *`feature = "value"`* -
/// The value type [`Val`](Val) can contain integers, floats, or bools.
///  
/// You can create a value as follows.
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::Val;
/// let v_f = Val::<i32, f64>::from_float(3.4);
/// let v_i = Val::<i32, f64>::from_int(3);
/// assert_eq!(v_f.to_float()?, 3.4);
/// assert_eq!(v_i.to_int()?, 3);
/// #
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Val<I = i32, F = f64>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    /// `Val`ues with can be created with [`from_float`](Val::from_float),
    /// [`from_int`](Val::from_int), or [`from_bool`](Val::from_bool).
    Int(I),
    Float(F),
    Bool(bool),
    /// Since the trait `Try` is experimental, we keep track of an error in an additional variant.
    Error(ExError),
    /// Sometimes, `Val` does not contain a value
    None,
}

impl<I, F> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    to_type!(to_int, I, Int);
    to_type!(to_float, F, Float);
    to_type!(to_bool, bool, Bool);

    from_type!(from_float, Float, F);
    from_type!(from_int, Int, I);
    from_type!(from_bool, Bool, bool);
}

fn map_parse_err<E: Debug>(e: E) -> ExError {
    ExError {
        msg: format!("{:?}", e),
    }
}

impl<I, F> FromStr for Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    type Err = ExError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let res = Ok(if s.contains(".") {
            Val::Float(s.parse::<F>().map_err(map_parse_err)?)
        } else if s == "false" || s == "true" {
            Val::Bool(s.parse::<bool>().map_err(map_parse_err)?)
        } else {
            Val::Int(s.parse::<I>().map_err(map_parse_err)?)
        });
        match res {
            Result::Ok(_) => res,
            Result::Err(e) => Err(ExError {
                msg: format!("could not parse {}, {:?}", s, e),
            }),
        }
    }
}

fn pow<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match (a, b) {
        (Val::Float(x), Val::Float(y)) => Val::<I, F>::from_float(x.powf(y)),
        (Val::Float(x), Val::Int(y)) => Val::<I, F>::from_float(x.powi(y.to_i32().unwrap())),
        (Val::Int(x), Val::Int(y)) => match y.to_u32() {
            Some(exponent_) => Val::<I, F>::from_int(x.pow(exponent_)),
            None => Val::Error(format_exerr!(
                "cannot convert {:?} to exponent of an int",
                y
            )),
        },
        _ => Val::Error(ExError::from_str("cannot compute power of")),
    }
}

macro_rules! base_arith {
    ($name:ident) => {
        fn $name<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            match (a, b) {
                (Val::Float(x), Val::Float(y)) => Val::<I, F>::from_float(x.$name(y)),
                (Val::Int(x), Val::Int(y)) => Val::<I, F>::from_int(x.$name(y)),
                _ => Val::Error(ExError::from_str(
                    format!("can only apply {} to 2 ints or 2 floats", stringify!($name)).as_str(),
                )),
            }
        }
    };
}

base_arith!(add);
base_arith!(sub);
base_arith!(mul);
base_arith!(div);

macro_rules! single_type_arith {
    ($name:ident, $variant:ident, $op:expr) => {
        fn $name<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            match (a, b) {
                (Val::$variant(na), Val::$variant(nb)) => $op(na, nb),
                _ => Val::Error(format_exerr!(
                    "can only apply 2 {}s to {}",
                    stringify!($variant),
                    stringify!($name)
                )),
            }
        }
    };
}

single_type_arith!(rem, Int, |a, b| Val::from_int(a % b));
single_type_arith!(bitwise_or, Int, |a, b| Val::from_int(a | b));
single_type_arith!(bitwise_and, Int, |a, b| Val::from_int(a & b));
single_type_arith!(bitwise_xor, Int, |a, b| Val::from_int(a ^ b));
single_type_arith!(right_shift, Int, |a: I, b: I| -> Val<I, F> {
    match b.to_usize() {
        Some(bu) => Val::from_int(a >> bu),
        None => Val::Error(format_exerr!("cannot convert {:?} to usize", b)),
    }
});
single_type_arith!(left_shift, Int, |a: I, b: I| -> Val<I, F> {
    match b.to_usize() {
        Some(bu) => Val::from_int(a << bu),
        None => Val::Error(format_exerr!("cannot convert {:?} to usize", b)),
    }
});

single_type_arith!(or, Bool, |a, b| Val::from_bool(a || b));
single_type_arith!(and, Bool, |a, b| Val::from_bool(a && b));

macro_rules! unary_match_name {
    ($name:ident, $scalar:ident, $(($unused_ops:expr, $variants:ident, $from_types:ident)),+) => {
        match $scalar {
            $(Val::$variants(x) => Val::<I, F>::$from_types(x.$name()),)+
            _ => Val::<I, F>::Error(format_exerr!("did not expect {:?}", $scalar)),
        }
    };
}

macro_rules! unary_match_op {
    ($name:ident, $scalar:ident, $(($ops:expr, $variants:ident, $unused_from_types:ident)),+) => {
        match $scalar {
            $(Val::$variants(x) =>  $ops(x),)+
            _ => Val::<I, F>::Error(format_exerr!("did not expect {:?}", $scalar)),
        }
    };
}

macro_rules! unary_match {
    ($name:ident, $matcher:ident, $(($ops:expr, $variants:ident, $from_types:ident)),+) => {
        fn $name<I, F>(val: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            $matcher!($name, val, $(($ops, $variants, $from_types)),+)
        }
    };
}

macro_rules! unary_name {
    ($name:ident, $(($variants:ident, $from_types:ident)),+) => {
        unary_match!($name, unary_match_name, $((0, $variants, $from_types)),+);
    }
}

unary_name!(abs, (Float, from_float), (Int, from_int));
unary_name!(signum, (Float, from_float), (Int, from_int));
unary_name!(sin, (Float, from_float));
unary_name!(cos, (Float, from_float));
unary_name!(tan, (Float, from_float));
unary_name!(asin, (Float, from_float));
unary_name!(acos, (Float, from_float));
unary_name!(atan, (Float, from_float));
unary_name!(sinh, (Float, from_float));
unary_name!(cosh, (Float, from_float));
unary_name!(tanh, (Float, from_float));
unary_name!(floor, (Float, from_float));
unary_name!(ceil, (Float, from_float));
unary_name!(trunc, (Float, from_float));
unary_name!(fract, (Float, from_float));
unary_name!(exp, (Float, from_float));
unary_name!(sqrt, (Float, from_float));
unary_name!(ln, (Float, from_float));
unary_name!(log2, (Float, from_float));
unary_name!(swap_bytes, (Int, from_int));
unary_name!(to_le, (Int, from_int));
unary_name!(to_be, (Int, from_int));

macro_rules! unary_op {
    ($name:ident, $(($ops:expr, $variants:ident)),+) => {
        unary_match!($name, unary_match_op, $(($ops, $variants, i32)),+);
    }
}

unary_op!(
    fact,
    (
        |a: I| if a == I::zero() {
            Val::from_int(I::one())
        } else {
            let a_usize_unpacked: usize = match a.to_usize() {
                Some(x) => x,
                None => return Val::Error(format_exerr!("cannot compute factorial of {:?}", a)),
            };
            let res = (1usize..(a_usize_unpacked + 1usize))
                .map(|i: usize| I::from(i))
                .fold(Some(I::one()), |a, b| match (a, b) {
                    (Some(a_), Some(b_)) => Some(a_ * b_),
                    _ => None,
                });
            match res {
                Some(i) => Val::from_int(i),
                None => Val::Error(format_exerr!("cannot compute factorial of {:?}", a)),
            }
        },
        Int
    )
);

unary_op!(
    minus,
    (|a: I| Val::from_int(-a), Int),
    (|a: F| Val::from_float(-a), Float)
);

/// *`feature = "value"`* - Factory of default operators for value data types.
///
/// Operators available in addition to those from [`FloatOpsFactory`](crate::FloatOpsFactory) are:
/// * `%` - reminder of integers
/// * `|` - bitwise or of integers
/// * `&` - bitwise and of integers
/// * `XOR` - bitwise xor of integers
/// * `<<` - left shift of integers
/// * `>>` - right shift of integers
/// * `||` - or for booleans
/// * `&&` - and for booleans
/// * `if` - returns first operand if second is true, else None, inspired by Python's ternary if-else-operator to `a if condition else b`,
/// * `else` - returns second operand if first is None, else first, inspired by Python's ternary if-else-operator to `a if condition else b`,
/// * `fact` - factorial of integers
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct ValOpsFactory<I = i32, F = f64>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    dummy_i: PhantomData<I>,
    dummy_f: PhantomData<F>,
}

impl<I, F> MakeOperators<Val<I, F>> for ValOpsFactory<I, F>
where
    I: DataType + PrimInt + Signed,
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
                    prio: 6,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: |a, b| add(a, b),
                    prio: 3,
                    is_commutative: true,
                },
            ),
            Operator::make_bin_unary(
                "-",
                BinOp {
                    apply: |a, b| sub(a, b),
                    prio: 3,
                    is_commutative: false,
                },
                |a| minus(a),
            ),
            Operator::make_bin(
                "*",
                BinOp {
                    apply: |a, b| mul(a, b),
                    prio: 4,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "/",
                BinOp {
                    apply: |a, b| div(a, b),
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "%",
                BinOp {
                    apply: |a, b| rem(a, b),
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "|",
                BinOp {
                    apply: |a, b| bitwise_or(a, b),
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "&",
                BinOp {
                    apply: |a, b| bitwise_and(a, b),
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "XOR",
                BinOp {
                    apply: |a, b| bitwise_xor(a, b),
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">>",
                BinOp {
                    apply: |a, b| right_shift(a, b),
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<<",
                BinOp {
                    apply: |a, b| left_shift(a, b),
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "&&",
                BinOp {
                    apply: |a, b| and(a, b),
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "||",
                BinOp {
                    apply: |a, b| or(a, b),
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "==",
                BinOp {
                    apply: |a, b| Val::from_bool(a == b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">=",
                BinOp {
                    apply: |a, b| Val::from_bool(a >= b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">",
                BinOp {
                    apply: |a, b| Val::from_bool(a > b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "<=",
                BinOp {
                    apply: |a, b| Val::from_bool(a <= b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "<",
                BinOp {
                    apply: |a, b| Val::from_bool(a < b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "!=",
                BinOp {
                    apply: |a, b| Val::from_bool(a != b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "if",
                BinOp {
                    apply: |v, cond| {
                        let condition = match cond.to_bool() {
                            Ok(b) => b,
                            Err(e) => return Val::Error(e),
                        };
                        if condition {
                            v
                        } else {
                            Val::None
                        }
                    },
                    prio: 0,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "else",
                BinOp {
                    apply: |res_of_if, v| match res_of_if {
                        Val::None => v,
                        _ => res_of_if,
                    },
                    prio: 0,
                    is_commutative: true,
                },
            ),
            Operator::make_unary("signum", |a| signum(a)),
            Operator::make_unary("abs", |a| abs(a)),
            Operator::make_unary("sin", |a| sin(a)),
            Operator::make_unary("cos", |a| cos(a)),
            Operator::make_unary("tan", |a| tan(a)),
            Operator::make_unary("asin", |a| asin(a)),
            Operator::make_unary("acos", |a| acos(a)),
            Operator::make_unary("atan", |a| atan(a)),
            Operator::make_unary("sinh", |a| sinh(a)),
            Operator::make_unary("cosh", |a| cosh(a)),
            Operator::make_unary("tanh", |a| tanh(a)),
            Operator::make_unary("floor", |a| floor(a)),
            Operator::make_unary("ceil", |a| ceil(a)),
            Operator::make_unary("trunc", |a| trunc(a)),
            Operator::make_unary("fract", |a| fract(a)),
            Operator::make_unary("exp", |a| exp(a)),
            Operator::make_unary("sqrt", |a| sqrt(a)),
            Operator::make_unary("log", |a| ln(a)),
            Operator::make_unary("log2", |a| log2(a)),
            Operator::make_unary("swap_bytes", |a| swap_bytes(a)),
            Operator::make_unary("to_le", |a| to_le(a)),
            Operator::make_unary("to_be", |a| to_be(a)),
            Operator::make_unary("fact", |a| fact(a)),
            Operator::make_constant(
                "PI",
                Val::from_float(F::from(std::f64::consts::PI).unwrap()),
            ),
            Operator::make_constant("π", Val::from_float(F::from(std::f64::consts::PI).unwrap())),
            Operator::make_constant("E", Val::from_float(F::from(std::f64::consts::E).unwrap())),
        ]
    }
}

/// *`feature = "value"`* - Parses a string into an expression of type
/// [`FlatExVal`](FlatExVal) with datatype [`Val`](Val).
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// # use exmex::{parse_val, Express, Val};
/// let expr = parse_val::<i32, f64>("x^y")?;
/// let res = expr.eval(&[Val::from_float(2.0), Val::from_int(3)])?.to_float()?;
/// assert!( (res - 8.0).abs() < 1e-12);
/// #
/// #     Ok(())
/// # }
/// ```
pub fn parse_val<I, F>(text: &str) -> ExResult<FlatEx<Val<I, F>, ValOpsFactory<I, F>>>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    FlatEx::<Val<I, F>, ValOpsFactory<I, F>>::from_regex(text, &RE_VAR_NAME_EXACT)
}

/// *`feature = "value"`* - Parses a string into an expression of type [`OwnedFlatExVal`](OwnedFlatExVal) with
/// datatype [`Val`](Val).
pub fn parse_val_owned<I, F>(text: &str) -> ExResult<OwnedFlatEx<Val<I, F>, ValOpsFactory<I, F>>>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    Ok(OwnedFlatEx::from_flatex(parse_val(text)?))
}

#[cfg(test)]
mod tests {

    use crate::{
        format_exerr,
        util::assert_float_eq_f64,
        value::{Val, RE_VAR_NAME_EXACT},
        ExError, ExResult, Express, FlatEx,
    };

    use super::ValOpsFactory;
    #[test]
    fn test_to() -> ExResult<()> {
        assert_eq!(Val::<i32, f64>::Float(3.4).to_float()?, 3.4);
        assert_eq!(Val::<i32, f64>::Int(123).to_int()?, 123);
        assert!(Val::<i32, f64>::Bool(true).to_bool()?);
        assert!(Val::<i32, f64>::Bool(false).to_int().is_err());
        assert_eq!(Val::<i32, f64>::Float(3.4).to_float()?, 3.4);
        assert_eq!(Val::<i32, f64>::Int(34).to_int()?, 34);
        assert!(!Val::<i32, f64>::Bool(false).to_bool()?);
        Ok(())
    }

    #[test]
    fn test_no_vars() -> ExResult<()> {
        fn test_int(s: &str, reference: i32) -> ExResult<()> {
            let res = FlatEx::<Val, ValOpsFactory>::from_regex(s, &RE_VAR_NAME_EXACT)?
                .eval(&[])?
                .to_int();
            match res {
                Ok(i) => {
                    assert_eq!(reference, i);
                }
                Err(e) => {
                    println!("{:?}", e);
                    assert!(false)
                }
            }
            Ok(())
        }
        fn test_float(s: &str, reference: f64) -> ExResult<()> {
            let expr = FlatEx::<Val, ValOpsFactory>::from_regex(s, &RE_VAR_NAME_EXACT)?;
            assert_float_eq_f64(reference, expr.eval(&[])?.to_float()?);
            Ok(())
        }
        fn test_bool(s: &str, reference: bool) -> ExResult<()> {
            let expr = FlatEx::<Val, ValOpsFactory>::from_regex(s, &RE_VAR_NAME_EXACT)?;
            assert_eq!(reference, expr.eval(&[])?.to_bool()?);
            Ok(())
        }
        fn test_error(s: &str) -> ExResult<()> {
            let expr = FlatEx::<Val, ValOpsFactory>::from_regex(s, &RE_VAR_NAME_EXACT);
            match expr {
                Ok(exp) => {
                    let v = exp.eval(&[])?;
                    match v {
                        Val::Error(e) => {
                            println!("found expected error {:?}", e);
                            Ok(())
                        }
                        _ => Err(format_exerr!("'{}' should fail but didn't", s)),
                    }
                }
                Err(e) => {
                    println!("found expected error {:?}", e);
                    Ok(())
                }
            }
        }
        fn test_none(s: &str) -> ExResult<()> {
            let expr = FlatEx::<Val, ValOpsFactory>::from_regex(s, &RE_VAR_NAME_EXACT)?;
            match expr.eval(&[])? {
                Val::None => Ok(()),
                _ => Err(format_exerr!("'{}' should return none but didn't", s)),
            }
        }
        test_float("2.0^2", 4.0)?;
        test_int("2^4", 16)?;
        test_error("2^-4")?;
        test_int("2+4", 6)?;
        test_int("9+4", 13)?;
        test_int("9+4^2", 25)?;
        test_int("9/4", 2)?;
        test_int("9%4", 1)?;
        test_float("2.5+4.0^2", 18.5)?;
        test_float("2.5*4.0^2", 2.5 * 4.0 * 4.0)?;
        test_float("2.5-4.0^-2", 2.5 - 4.0f64.powi(-2))?;
        test_float("9.0/4.0", 9.0 / 4.0)?;
        test_float("sin(9.0)", 9.0f64.sin())?;
        test_float("cos(91.0)", 91.0f64.cos())?;
        test_float("tan(913.0)", 913.0f64.tan())?;
        test_float("sin(-π)", 0.0)?;
        test_float("cos(π)", -1.0)?;
        test_float("sin (1 if false else 2.0)", 2.0f64.sin())?;
        test_int("1 if true else 2.0", 1)?;
        test_float("(9.0 if true else 2.0)", 9.0)?;
        test_int("1<<4-2", 4)?;
        test_int("4>>2", 1)?;
        test_int("signum(4>>1)", 1)?;
        test_float("signum(-123.12)", -1.0)?;
        test_float("abs(-123.12)", 123.12)?;
        test_int("fact(4)", 2 * 3 * 4)?;
        test_int("fact(0)", 1)?;
        test_error("fact(-1)")?;
        test_bool("1>2", false)?;
        test_bool("1<2", true)?;
        test_bool("1.4>=1.4", true)?;
        test_bool("true==true", true)?;
        test_bool("false==true", false)?;
        test_bool("1.5 != 1.5 + 2.0", true)?;
        test_error("1 + 1.0")?;
        test_bool("1.0 == 1", false)?;
        test_bool("1 == 1", true)?;
        test_bool("true else 2", true)?;
        test_int("1 else 2", 1)?;
        test_error("if true else 2")?;
        test_none("2 if false")?;
        Ok(())
    }
}
