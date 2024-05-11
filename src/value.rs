use std::{cmp::Ordering, fmt::Debug, marker::PhantomData, str::FromStr};

use num::{Float, PrimInt, Signed};
use smallvec::{smallvec, SmallVec};

use crate::{
    data_type::DataType, exerr, expression::MatchLiteral, literal_matcher_from_pattern,
    result::to_ex, BinOp, ExError, ExResult, Express, FlatEx, MakeOperators, Operator,
};

pub type ArrayType<F> = SmallVec<[F; 4]>;

/// *`feature = "value"`* -
/// The value type [`Val`](Val) can contain an integer, float, bool, a vector of floats, none, or error.
/// To use the value type, there are the is a parse function [`parse_val`](`parse_val`).
/// In the following example, the ternary Python-style `a if condition else b` is used.
/// This is equivalent to `if condition {a} else {b}` in Rust or `condition ? a : b` in C.
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{Express, Val};
/// let expr = exmex::parse_val::<i32, f64>("1.0 if x > y else 73")?;
/// assert_eq!(expr.eval(&[Val::Float(3.4), Val::Int(3)])?.to_float()?, 1.0);
/// assert_eq!(expr.eval(&[Val::Int(34), Val::Float(132.0)])?.to_int()?, 73);
/// #
/// #     Ok(())
/// # }
/// ```
/// Note that the ternary operator is actually implemented as two binary operators called `if` and `else`.
/// To this end, we return `Val::None` from the `if`-operator if and only if the condition is false. On the flipside,
/// this has strange side effects such as `5 else 3` being a valid expression evaluating to 5.   
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::Express;
/// let expr = exmex::parse_val::<i32, f64>("5 else 3")?;
/// assert_eq!(expr.eval(&[])?.to_int()?, 5);
/// #
/// #     Ok(())
/// # }
/// ```
/// We use the variant `Error` to report errors, since the trait `Try` is not yet stable.
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::Express;
/// let expr = exmex::parse_val::<i32, f64>("fact(3.5)")?;
/// let res = expr.eval(&[])?;
/// assert!(format!("{:?}", res) == "Error(ExError { msg: \"did not expect Float(3.5)\" })");
/// #
/// #     Ok(())
/// # }
/// ```
/// When converting the value to the expected primitive type with `to_int`, `to_float`, or `to_bool`, the case `Val::Error(ExError)` is
/// converted to `ExResult::Err(ExError)`.
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// # use exmex::Express;
/// # let expr = exmex::parse_val::<i32, f64>("fact(3.5)")?;
/// # let res = expr.eval(&[])?;
/// # assert!(format!("{:?}", res) == "Error(ExError { msg: \"did not expect Float(3.5)\" })");
/// assert!(res.to_int().is_err());
/// #
/// #     Ok(())
/// # }
/// ```
///
/// An example with vectors is shown in the following.
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// # use exmex::Express;
/// use exmex::Val;
/// use smallvec::smallvec;
///
/// // dot product of two vectors, one as parameter the other as literal
/// let expr = exmex::parse_val::<i32, f64>("dot(v, [1, 0, 0])")?;
/// let v = Val::Array(smallvec![3.0, 4.0, 2.0]);
/// let res = expr.eval(&[v])?;
/// assert!(res.to_float()? == 3.0);
///
/// // The following example shows how to get the second component of a vector
/// let expr = exmex::parse_val::<i32, f64>("(v + [1, 0, 0]).2")?;
/// let v = Val::Array(smallvec![3.0, 4.0, 2.0]);
/// let res = expr.eval(&[v])?;
/// assert!(res.to_float()? == 2.0);
/// #
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub enum Val<I = i32, F = f64>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    Array(ArrayType<F>),
    Int(I),
    Float(F),
    Bool(bool),
    /// Since the trait `Try` is experimental, we keep track of an error in an additional variant.
    Error(ExError),
    /// Sometimes, `Val` does not contain a value
    #[default]
    None,
}

impl<I, F> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    pub fn to_bool(self) -> ExResult<bool> {
        match self {
            Self::Bool(b) => Ok(b),
            Self::Int(n) => Ok(n != I::from(0).unwrap()),
            Self::Float(x) => Ok(x != F::from(0.0).unwrap()),
            Self::Error(e) => Err(e),
            Self::Array(a) => Err(exerr!(
                "array is not a scalar and hence not a bool, {:?}",
                a
            )),
            Self::None => Err(ExError::new(
                "`Val` of `Val::None` cannot be converted to float",
            )),
        }
    }
    pub fn to_float(self) -> ExResult<F> {
        match self {
            Self::Bool(b) => Ok(F::from(if b { 1.0 } else { 0.0 }).unwrap()),
            Self::Int(n) => F::from(n).ok_or_else(|| exerr!("cannot convert {n:?} to float")),
            Self::Float(x) => Ok(x),
            Self::Error(e) => Err(e),
            Self::Array(a) => Err(exerr!("array is not a scalar and hence not a float, {a:?}")),
            Self::None => Err(ExError::new(
                "`Val` of `Val::None` cannot be converted to float",
            )),
        }
    }
    pub fn to_int(self) -> ExResult<I> {
        match self {
            Self::Bool(b) => Ok(I::from(if b { 1 } else { 0 }).unwrap()),
            Self::Float(x) => I::from(x).ok_or_else(|| exerr!("cannot convert {x:?} to int")),
            Self::Int(n) => Ok(n),
            Self::Error(e) => Err(e),
            Self::Array(a) => Err(exerr!("array is not a scalar and hence not a int, {a:?}")),
            Self::None => Err(ExError::new(
                "`Val` of `Val::None` cannot be converted to float",
            )),
        }
    }
    pub fn to_float_val(self) -> Self {
        match self.to_float() {
            Ok(f) => Val::Float(f),
            Err(e) => Val::Error(e),
        }
    }
    pub fn to_array(self) -> ExResult<ArrayType<F>> {
        match self {
            Self::Array(a) => Ok(a),
            _ => Err(exerr!("cannot convert {self:?} to array")),
        }
    }
}

impl<I, F> From<f32> for Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    fn from(value: f32) -> Self {
        Val::Float(F::from(value).unwrap())
    }
}

impl<I, F> From<u8> for Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    fn from(value: u8) -> Self {
        Val::Int(I::from(value).unwrap())
    }
}

fn try_parse<F>(s: &str) -> ExResult<F>
where
    F: DataType + Float,
    <F as FromStr>::Err: Debug,
{
    match s.parse::<F>() {
        Ok(f) => Ok(f),
        Err(e) => Err(exerr!("could not parse '{s:?}' as float due to {e:?}")),
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
        Ok(if s.contains('[') {
            let s = s.trim_start_matches('[').trim_end_matches(']');
            Val::Array(
                s.split(',')
                    .map(|xi| try_parse(xi.trim()))
                    .collect::<ExResult<ArrayType<F>>>()?,
            )
        } else if s.contains('.') {
            Val::Float(s.parse::<F>().map_err(to_ex)?)
        } else if s == "false" || s == "true" {
            Val::Bool(s.parse::<bool>().map_err(to_ex)?)
        } else {
            Val::Int(s.parse::<I>().map_err(to_ex)?)
        })
    }
}

impl<I, F> PartialEq<Val<I, F>> for Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    fn eq(&self, other: &Val<I, F>) -> bool {
        match (self, other) {
            (Val::Float(x), Val::Float(y)) => x == y,
            (Val::Int(x), Val::Int(y)) => x == y,
            (Val::Bool(x), Val::Bool(y)) => x == y,
            (Val::Float(x), Val::Int(y)) => *x == F::from(*y).unwrap(),
            (Val::Int(x), Val::Float(y)) => F::from(*x).unwrap() == *y,
            _ => false,
        }
    }
}

impl<I, F> PartialOrd<Val<I, F>> for Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    fn partial_cmp(&self, other: &Val<I, F>) -> Option<Ordering> {
        match (self, other) {
            (Val::Float(x), Val::Float(y)) => x.partial_cmp(y),
            (Val::Int(x), Val::Int(y)) => x.partial_cmp(y),
            (Val::Float(x), Val::Int(y)) => x.partial_cmp(&F::from(*y).unwrap()),
            (Val::Int(x), Val::Float(y)) => F::from(*x).unwrap().partial_cmp(y),
            _ => None,
        }
    }
}

fn pow<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match (a, b) {
        (Val::Float(x), Val::Float(y)) => Val::Float(x.powf(y)),
        (Val::Float(x), Val::Int(y)) => Val::Float(x.powi(y.to_i32().unwrap())),
        (Val::Int(x), Val::Int(y)) => match y.to_usize() {
            Some(exponent_) => match num::checked_pow(x, exponent_) {
                Some(res) => Val::Int(res),
                None => Val::Error(exerr!("overflow in {:?}^{:?}", x, y)),
            },
            None => Val::Error(exerr!("cannot convert {:?} to exponent of an int", y)),
        },
        (Val::Error(e), _) => Val::Error(e),
        (_, Val::Error(e)) => Val::Error(e),
        _ => Val::Error(exerr!("cannot compute power",)),
    }
}

macro_rules! base_arith {
    ($name:ident, $intname:ident,$accessint:expr, $wrapint:expr) => {
        fn $name<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            match (a, b) {
                (Val::Float(x), Val::Float(y)) => Val::Float(x.$name(y)),
                (Val::Float(y), Val::Array(x)) => {
                    Val::Array(x.iter().map(|xi| xi.$name(y)).collect())
                }
                (Val::Array(x), Val::Float(y)) => {
                    Val::Array(x.iter().map(|xi| xi.$name(y)).collect())
                }
                (Val::Int(y), Val::Array(x)) => {
                    Val::Array(x.iter().map(|xi| xi.$name(F::from(y).unwrap())).collect())
                }
                (Val::Array(x), Val::Int(y)) => {
                    Val::Array(x.iter().map(|xi| xi.$name(F::from(y).unwrap())).collect())
                }
                (Val::Array(x), Val::Array(y)) => Val::Array(
                    x.iter()
                        .zip(y.iter())
                        .map(|(xi, yi)| xi.$name(*yi))
                        .collect(),
                ),
                (Val::Int(x), Val::Int(y)) => match $wrapint(x.$intname($accessint(&y))) {
                    Some(res) => Val::Int(res),
                    None => Val::Error(exerr!(
                        "overflow in {:?}{:?}{:?}",
                        x,
                        stringify!($intname),
                        y
                    )),
                },
                (Val::Float(x), Val::Int(y)) => Val::Float(x.$name(F::from(y).unwrap())),
                (Val::Int(x), Val::Float(y)) => Val::Float(F::from(x).unwrap().$name(y)),
                (Val::Error(e), _) => Val::Error(e),
                (_, Val::Error(e)) => Val::Error(e),
                _ => Val::Error(ExError::new(
                    format!("can only apply {} to ints or floats", stringify!($name)).as_str(),
                )),
            }
        }
    };
}

base_arith!(add, checked_add, |x| x, |x| x);
base_arith!(sub, checked_sub, |x| x, |x| x);
base_arith!(mul, checked_mul, |x| x, |x| x);
base_arith!(div, checked_div, |x| x, |x| x);
base_arith!(min, min, |x: &I| *x, |x| Some(x));
base_arith!(max, max, |x: &I| *x, |x| Some(x));

macro_rules! single_type_arith {
    ($name:ident, $variant:ident, $op:expr) => {
        fn $name<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            // https://github.com/rust-lang/rust-clippy/issues/11274
            #[allow(clippy::redundant_closure_call)]
            match (a, b) {
                (Val::$variant(na), Val::$variant(nb)) => $op(na, nb),
                (Val::Error(e), _) => Val::Error(e),
                (_, Val::Error(e)) => Val::Error(e),
                _ => Val::Error(exerr!(
                    "can only apply 2 {}s to {}",
                    stringify!($variant),
                    stringify!($name)
                )),
            }
        }
    };
}

single_type_arith!(rem, Int, |a, b| if b == I::zero() {
    Val::Error(ExError::new("% by zero"))
} else {
    Val::Int(a % b)
});
single_type_arith!(bitwise_or, Int, |a, b| Val::Int(a | b));
single_type_arith!(bitwise_and, Int, |a, b| Val::Int(a & b));
single_type_arith!(bitwise_xor, Int, |a, b| Val::Int(a ^ b));
single_type_arith!(right_shift, Int, |a: I, b: I| -> Val<I, F> {
    match b.to_usize() {
        Some(bu) if b.to_usize().unwrap() < (a.count_ones() + a.count_zeros()) as usize => {
            Val::Int(a >> bu)
        }
        _ => Val::Error(exerr!("cannot shift right {:?} by {:?}", a, b)),
    }
});
single_type_arith!(left_shift, Int, |a: I, b: I| -> Val<I, F> {
    match b.to_usize() {
        Some(bu) if b.to_usize().unwrap() < (a.count_ones() + a.count_zeros()) as usize => {
            Val::Int(a << bu)
        }
        _ => Val::Error(exerr!("cannot shift left {:?} by {:?}", a, b)),
    }
});

fn and<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    <I as FromStr>::Err: Debug,
    F: DataType + Float,
    <F as FromStr>::Err: Debug,
{
    match (&a, &b) {
        (Val::Bool(a), Val::Bool(b)) => Val::Bool(*a && *b),
        _ => {
            if a.clone() <= b.clone() {
                a
            } else {
                b
            }
        }
    }
}
fn or<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    <I as FromStr>::Err: Debug,
    F: DataType + Float,
    <F as FromStr>::Err: Debug,
{
    match (&a, &b) {
        (Val::Bool(a), Val::Bool(b)) => Val::Bool(*a || *b),
        _ => {
            if a >= b {
                a
            } else {
                b
            }
        }
    }
}
fn atan2<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    <I as FromStr>::Err: Debug,
    F: DataType + Float,
    <F as FromStr>::Err: Debug,
{
    let a = a.to_float_val();
    let b = b.to_float_val();
    match (a, b) {
        (Val::Float(a), Val::Float(b)) => Val::Float(a.atan2(b)),
        (_, Val::Error(e)) => Val::Error(e),
        (Val::Error(e), _) => Val::Error(e),
        _ => Val::Error(exerr!("could not apply atan2 to",)),
    }
}
macro_rules! unary_match_name {
    ($name:ident, $scalar:ident, $(($unused_ops:expr, $variants:ident)),+) => {
        match $scalar {
            $(Val::$variants(x) => Val::$variants(x.$name()),)+
            Val::Error(_) => $scalar,
            _ => Val::<I, F>::Error(exerr!("did not expect {:?}", $scalar)),
        }
    };
}

macro_rules! unary_match_op {
    ($name:ident, $scalar:ident, $(($ops:expr, $variants:ident)),+) => {
        // https://github.com/rust-lang/rust-clippy/issues/11274
        #[allow(clippy::redundant_closure_call)]
        match $scalar {
            $(Val::$variants(x) => $ops(x),)+
            Val::Error(_) => $scalar,
            _ => Val::<I, F>::Error(exerr!("did not expect {:?}", $scalar)),
        }
    };
}

macro_rules! unary_match {
    ($name:ident, $matcher:ident, $(($ops:expr, $variants:ident)),+) => {
        fn $name<I, F>(val: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            $matcher!($name, val, $(($ops, $variants)),+)
        }
    };
}

macro_rules! unary_name {
    ($name:ident, $($variants:ident),+) => {
        unary_match!($name, unary_match_name, $((0, $variants)),+);
    }
}

unary_name!(abs, Float, Int);
unary_name!(signum, Float, Int);
unary_name!(sin, Float);
unary_name!(round, Float);
unary_name!(cos, Float);
unary_name!(tan, Float);
unary_name!(asin, Float);
unary_name!(acos, Float);
unary_name!(atan, Float);
unary_name!(sinh, Float);
unary_name!(cosh, Float);
unary_name!(tanh, Float);
unary_name!(asinh, Float);
unary_name!(acosh, Float);
unary_name!(atanh, Float);
unary_name!(floor, Float);
unary_name!(ceil, Float);
unary_name!(trunc, Float);
unary_name!(fract, Float);
unary_name!(exp, Float);
unary_name!(sqrt, Float);
unary_name!(cbrt, Float);
unary_name!(ln, Float);
unary_name!(log2, Float);
unary_name!(log10, Float);
unary_name!(swap_bytes, Int);
unary_name!(to_le, Int);
unary_name!(to_be, Int);

macro_rules! unary_op {
    ($name:ident, $(($ops:expr, $variants:ident)),+) => {
        unary_match!($name, unary_match_op, $(($ops, $variants)),+);
    }
}

unary_op!(
    fact,
    (
        |a: I| if a == I::zero() {
            Val::Int(I::one())
        } else {
            let a_usize_unpacked: usize = match a.to_usize() {
                Some(x) => x,
                None => return Val::Error(exerr!("cannot compute factorial of {:?}", a)),
            };
            let res = (1usize..(a_usize_unpacked + 1usize))
                .map(I::from)
                .try_fold(I::one(), |a, b| b.and_then(|b| a.checked_mul(&b)));
            match res {
                Some(i) => Val::Int(i),
                None => Val::Error(exerr!("cannot compute factorial of {:?}", a)),
            }
        },
        Int
    )
);

unary_op!(
    minus,
    (|a: I| Val::Int(-a), Int),
    (|a: F| Val::Float(-a), Float),
    (
        |a: ArrayType<F>| Val::Array(a.iter().map(|ai| -(*ai)).collect()),
        Array
    )
);

macro_rules! cast {
    ($name:ident, $variant:ident, $other_variant:ident, $T:ident) => {
        fn $name<I, F>(v: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
            <I as FromStr>::Err: Debug,
            <F as FromStr>::Err: Debug,
        {
            match v {
                Val::$variant(x) => Val::$variant(x),
                Val::$other_variant(x) => Val::$variant($T::from(x).unwrap()),
                Val::Bool(x) => Val::$variant(if x { $T::one() } else { $T::zero() }),
                _ => Val::Error(exerr!("cannot convert '{:?}' to float", v)),
            }
        }
    };
}

cast!(cast_to_float, Float, Int, F);
cast!(cast_to_int, Int, Float, I);

fn dot<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match (a, b) {
        (Val::Array(a), Val::Array(b)) => {
            if a.len() != b.len() {
                return Val::Error(exerr!(
                    "cannot compute dot product of arrays of different lengths"
                ));
            }
            Val::Float(
                a.iter()
                    .zip(b.iter())
                    .map(|(ai, bi)| *ai * *bi)
                    .fold(F::zero(), |acc, x| acc + x),
            )
        }
        (Val::Error(e), _) => Val::Error(e),
        (_, Val::Error(e)) => Val::Error(e),
        _ => Val::Error(exerr!("cannot compute dot product")),
    }
}

fn length<I, F>(a: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match dot(a.clone(), a) {
        Val::Float(x) => Val::Float(x.sqrt()),
        Val::Error(e) => Val::Error(e),
        _ => Val::Error(exerr!(
            "cannot compute length, result of dot should be float"
        )),
    }
}

fn cross<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match (a, b) {
        (Val::Array(a), Val::Array(b)) => {
            if a.len() != 3 || b.len() != 3 {
                return Val::Error(exerr!(
                    "cannot compute cross product of arrays of different lengths"
                ));
            }
            let x = smallvec![
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            ];
            Val::Array(x)
        }
        (Val::Error(e), _) => Val::Error(e),
        (_, Val::Error(e)) => Val::Error(e),
        _ => Val::Error(exerr!("cannot compute cross product")),
    }
}

fn component<I, F>(a: Val<I, F>, i: Val<I, F>) -> Val<I, F>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
    match (a, i) {
        (Val::Array(a), Val::Int(i)) => {
            let len_i = I::from(a.len());
            if len_i.is_none() || len_i.map(|len| len <= i) == Some(true) || i < I::from(0).unwrap() {
                return Val::Error(exerr!(
                    "array has length {} but index {i:?} is requested",
                    a.len()
                ));
            }
            Val::Float(a[i.to_usize().unwrap()])
        }
        (Val::Error(e), _) => Val::Error(e),
        (_, Val::Error(e)) => Val::Error(e),
        _ => Val::Error(exerr!("to get the component of an array, the first argument must be an array and the second an integer")),
    }
}

/// *`feature = "value"`* - Factory of default operators for the data type [`Val`](Val).
///
/// Operators available in addition to those from [`FloatOpsFactory`](crate::FloatOpsFactory) are:
///
/// |representation|description|
/// |--------------|-----------|
/// | `%` | reminder of integers |
/// | <code>&#124;</code> | bitwise or of integers |
/// | `&` | bitwise and of integers |
/// | `XOR` | bitwise exclusive or of integers |
/// | `<<` | left shift of integers |
/// | `>>` | right shift of integers |
/// | <code>&#124;&#124;</code> | or for booleans |
/// | `&&` | and for booleans |
/// | `if` | returns first operand if second is true, else `Val::None`, to make `x if condition else y` possible |
/// | `else` | returns second operand if first is `Val::None`, else first, to make `x if condition else y` possible |
/// | `==`, `!=`, `<`, `>`, `<=`, `>=`| comparison operators between numbers, e.g., `1 == 1.0` is true. Comparing booleans to none-booleans is false, e.g., `1 == true` is false. Comparisons with `Val::None` or `Val::Error` always results in `false`, e.g., `(5 if false) == (5 if false)` is false.|
/// | `fact` | factorial of integers |
/// | `to_float` | convert integer, float, or bool to float |
/// | `to_int` | convert integer, float, or bool to integer |
///
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
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
                    apply: pow,
                    prio: 6,
                    is_commutative: false,
                },
            ),
            Operator::make_bin_unary(
                "+",
                BinOp {
                    apply: add,
                    prio: 3,
                    is_commutative: true,
                },
                |x| x,
            ),
            Operator::make_bin_unary(
                "-",
                BinOp {
                    apply: sub,
                    prio: 3,
                    is_commutative: false,
                },
                minus,
            ),
            Operator::make_bin(
                "cross",
                BinOp {
                    apply: cross,
                    prio: 4,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "dot",
                BinOp {
                    apply: dot,
                    prio: 4,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "*",
                BinOp {
                    apply: mul,
                    prio: 4,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "/",
                BinOp {
                    apply: |a, b| match b {
                        Val::Int(x) if x == I::zero() => {
                            Val::Error(ExError::new("int division by zero"))
                        }
                        _ => div(a, b),
                    },
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "atan2",
                BinOp {
                    apply: atan2,
                    prio: 0,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "%",
                BinOp {
                    apply: rem,
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "|",
                BinOp {
                    apply: bitwise_or,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "&",
                BinOp {
                    apply: bitwise_and,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "XOR",
                BinOp {
                    apply: bitwise_xor,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">>",
                BinOp {
                    apply: right_shift,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<<",
                BinOp {
                    apply: left_shift,
                    prio: 2,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "&&",
                BinOp {
                    apply: and,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "||",
                BinOp {
                    apply: or,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "==",
                BinOp {
                    apply: |a, b| Val::Bool(a == b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">=",
                BinOp {
                    apply: |a, b| Val::Bool(a >= b),
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ">",
                BinOp {
                    apply: |a, b| Val::Bool(a > b),
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<=",
                BinOp {
                    apply: |a, b| Val::Bool(a <= b),
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "<",
                BinOp {
                    apply: |a, b| Val::Bool(a < b),
                    prio: 1,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "!=",
                BinOp {
                    apply: |a, b| Val::Bool(a != b),
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
                    is_commutative: false,
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
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "min",
                BinOp {
                    apply: |x, y| min(x, y),
                    prio: 0,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "max",
                BinOp {
                    apply: |x, y| max(x, y),
                    prio: 0,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                ".",
                BinOp {
                    apply: |x, i| component(x, i),
                    prio: 5,
                    is_commutative: false,
                },
            ),
            Operator::make_unary("signum", signum),
            Operator::make_unary("abs", abs),
            Operator::make_unary("sin", sin),
            Operator::make_unary("cos", cos),
            Operator::make_unary("tan", tan),
            Operator::make_unary("asin", asin),
            Operator::make_unary("acos", acos),
            Operator::make_unary("atan", atan),
            Operator::make_unary("sinh", sinh),
            Operator::make_unary("cosh", cosh),
            Operator::make_unary("tanh", tanh),
            Operator::make_unary("asinh", asinh),
            Operator::make_unary("acosh", acosh),
            Operator::make_unary("atanh", atanh),
            Operator::make_unary("floor", floor),
            Operator::make_unary("ceil", ceil),
            Operator::make_unary("trunc", trunc),
            Operator::make_unary("fract", fract),
            Operator::make_unary("exp", exp),
            Operator::make_unary("sqrt", sqrt),
            Operator::make_unary("cbrt", cbrt),
            Operator::make_unary("round", round),
            Operator::make_unary("ln", ln),
            Operator::make_unary("log10", log10),
            Operator::make_unary("log2", log2),
            Operator::make_unary("log", ln),
            Operator::make_unary("swap_bytes", swap_bytes),
            Operator::make_unary("to_le", to_le),
            Operator::make_unary("to_be", to_be),
            Operator::make_unary("fact", fact),
            Operator::make_unary("to_int", cast_to_int),
            Operator::make_unary("to_float", cast_to_float),
            Operator::make_unary("length", length),
            Operator::make_constant("PI", Val::Float(F::from(std::f64::consts::PI).unwrap())),
            Operator::make_constant("π", Val::Float(F::from(std::f64::consts::PI).unwrap())),
            Operator::make_constant("E", Val::Float(F::from(std::f64::consts::E).unwrap())),
            Operator::make_constant("TAU", Val::Float(F::from(std::f64::consts::TAU).unwrap())),
            Operator::make_constant("τ", Val::Float(F::from(std::f64::consts::TAU).unwrap())),
        ]
    }
}
const PATTERN: &str = r"^([0-9]+(\.[0-9]+)?|true|false|\[\s*(\-?.?[0-9]+(\.[0-9]+)?|true|false)(\s*,\s*-?\.?[0-9]+(\.[0-9]+)?|true|false)*\s*\])";
literal_matcher_from_pattern!(ValMatcher, PATTERN);

/// *`feature = "value"`* - Alias for [`FlatEx`](FlatEx) with [`Val`](Val) as data type and [`ValOpsFactory`](ValOpsFactory)
/// as operator factory.
pub type FlatExVal<I, F> = FlatEx<Val<I, F>, ValOpsFactory<I, F>, ValMatcher>;

/// *`feature = "value"`* - Parses a string into an expression of type
/// [`FlatExVal`](FlatExVal) with datatype [`Val`](Val).
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// # use exmex::{parse_val, Express, Val};
/// let expr = parse_val::<i32, f64>("x^y")?;
/// let res = expr.eval(&[Val::Float(2.0), Val::Int(3)])?.to_float()?;
/// assert!( (res - 8.0).abs() < 1e-12);
/// #
/// #     Ok(())
/// # }
/// ```
pub fn parse_val<I, F>(text: &str) -> ExResult<FlatExVal<I, F>>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    FlatEx::<Val<I, F>, ValOpsFactory<I, F>, ValMatcher>::parse(text)
}
