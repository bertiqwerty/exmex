use std::{cmp::Ordering, fmt::Debug, marker::PhantomData, str::FromStr};

use num::{Float, PrimInt, Signed};

use crate::{
    data_type::DataType, expression::MatchLiteral, format_exerr, literal_matcher_from_pattern,
    BinOp, ExError, ExResult, FlatEx, MakeOperators, Operator,
};

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

/// *`feature = "value"`* -
/// The value type [`Val`](Val) can contain an integer, float, bool, none, or error.
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
#[derive(Clone, Debug)]
pub enum Val<I = i32, F = f64>
where
    I: DataType + PrimInt + Signed,
    F: DataType + Float,
{
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
        let res = Ok(if s.contains('.') {
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
                None => Val::Error(format_exerr!("overflow in {:?}^{:?}", x, y)),
            },
            None => Val::Error(format_exerr!(
                "cannot convert {:?} to exponent of an int",
                y
            )),
        },
        _ => Val::Error(ExError::new("cannot compute power of")),
    }
}

macro_rules! base_arith {
    ($name:ident, $intname:ident) => {
        fn $name<I, F>(a: Val<I, F>, b: Val<I, F>) -> Val<I, F>
        where
            I: DataType + PrimInt + Signed,
            F: DataType + Float,
        {
            match (a, b) {
                (Val::Float(x), Val::Float(y)) => Val::Float(x.$name(y)),
                (Val::Int(x), Val::Int(y)) => match x.$intname(&y) {
                    Some(res) => Val::Int(res),
                    None => Val::Error(format_exerr!(
                        "overflow in {:?}{:?}{:?}",
                        x,
                        stringify!($intname),
                        y
                    )),
                },
                (Val::Float(x), Val::Int(y)) => Val::Float(x.$name(F::from(y).unwrap())),
                (Val::Int(x), Val::Float(y)) => Val::Float(F::from(x).unwrap().$name(y)),
                _ => Val::Error(ExError::new(
                    format!("can only apply {} to ints or floats", stringify!($name)).as_str(),
                )),
            }
        }
    };
}

base_arith!(add, checked_add);
base_arith!(sub, checked_sub);
base_arith!(mul, checked_mul);
base_arith!(div, checked_div);

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
        _ => Val::Error(format_exerr!("cannot shift right {:?} by {:?}", a, b)),
    }
});
single_type_arith!(left_shift, Int, |a: I, b: I| -> Val<I, F> {
    match b.to_usize() {
        Some(bu) if b.to_usize().unwrap() < (a.count_ones() + a.count_zeros()) as usize => {
            Val::Int(a << bu)
        }
        _ => Val::Error(format_exerr!("cannot shift left {:?} by {:?}", a, b)),
    }
});

single_type_arith!(or, Bool, |a, b| Val::Bool(a || b));
single_type_arith!(and, Bool, |a, b| Val::Bool(a && b));

macro_rules! unary_match_name {
    ($name:ident, $scalar:ident, $(($unused_ops:expr, $variants:ident)),+) => {
        match $scalar {
            $(Val::$variants(x) => Val::$variants(x.$name()),)+
            _ => Val::<I, F>::Error(format_exerr!("did not expect {:?}", $scalar)),
        }
    };
}

macro_rules! unary_match_op {
    ($name:ident, $scalar:ident, $(($ops:expr, $variants:ident)),+) => {
        match $scalar {
            $(Val::$variants(x) =>  $ops(x),)+
            _ => Val::<I, F>::Error(format_exerr!("did not expect {:?}", $scalar)),
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
unary_name!(floor, Float);
unary_name!(ceil, Float);
unary_name!(trunc, Float);
unary_name!(fract, Float);
unary_name!(exp, Float);
unary_name!(sqrt, Float);
unary_name!(cbrt, Float);
unary_name!(ln, Float);
unary_name!(log2, Float);
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
                None => return Val::Error(format_exerr!("cannot compute factorial of {:?}", a)),
            };
            let res =
                (1usize..(a_usize_unpacked + 1usize))
                    .map(I::from)
                    .fold(Some(I::one()), |a, b| match (a, b) {
                        (Some(a_), Some(b_)) => Some(a_ * b_),
                        _ => None,
                    });
            match res {
                Some(i) => Val::Int(i),
                None => Val::Error(format_exerr!("cannot compute factorial of {:?}", a)),
            }
        },
        Int
    )
);

unary_op!(
    minus,
    (|a: I| Val::Int(-a), Int),
    (|a: F| Val::Float(-a), Float)
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
                _ => Val::Error(format_exerr!("cannot convert '{:?}' to float", v)),
            }
        }
    };
}

cast!(cast_to_float, Float, Int, F);
cast!(cast_to_int, Int, Float, I);

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
                    apply: pow,
                    prio: 6,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "+",
                BinOp {
                    apply: add,
                    prio: 3,
                    is_commutative: true,
                },
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
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                ">",
                BinOp {
                    apply: |a, b| Val::Bool(a > b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "<=",
                BinOp {
                    apply: |a, b| Val::Bool(a <= b),
                    prio: 1,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "<",
                BinOp {
                    apply: |a, b| Val::Bool(a < b),
                    prio: 1,
                    is_commutative: true,
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
            Operator::make_unary("floor", floor),
            Operator::make_unary("ceil", ceil),
            Operator::make_unary("trunc", trunc),
            Operator::make_unary("fract", fract),
            Operator::make_unary("exp", exp),
            Operator::make_unary("sqrt", sqrt),
            Operator::make_unary("cbrt", cbrt),
            Operator::make_unary("round", round),
            Operator::make_unary("log", ln),
            Operator::make_unary("log2", log2),
            Operator::make_unary("swap_bytes", swap_bytes),
            Operator::make_unary("to_le", to_le),
            Operator::make_unary("to_be", to_be),
            Operator::make_unary("fact", fact),
            Operator::make_unary("to_int", cast_to_int),
            Operator::make_unary("to_float", cast_to_float),
            Operator::make_constant("PI", Val::Float(F::from(std::f64::consts::PI).unwrap())),
            Operator::make_constant("π", Val::Float(F::from(std::f64::consts::PI).unwrap())),
            Operator::make_constant("E", Val::Float(F::from(std::f64::consts::E).unwrap())),
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
    FlatEx::<Val<I, F>, ValOpsFactory<I, F>, ValMatcher>::from_str(text)
}

#[cfg(test)]
mod tests {
    use crate::{
        format_exerr, parse_val, util::assert_float_eq_f64, value::Val, ExError, ExResult, Express,
        FlatExVal,
    };
    use std::str::FromStr;

    #[test]
    fn test_to() -> ExResult<()> {
        assert_float_eq_f64(Val::<i32, f64>::Float(3.4).to_float()?, 3.4);
        assert_eq!(Val::<i32, f64>::Int(123).to_int()?, 123);
        assert!(Val::<i32, f64>::Bool(true).to_bool()?);
        assert!(Val::<i32, f64>::Bool(false).to_int().is_err());
        assert_float_eq_f64(Val::<i32, f64>::Float(3.4).to_float()?, 3.4);
        assert_eq!(Val::<i32, f64>::Int(34).to_int()?, 34);
        assert!(!Val::<i32, f64>::Bool(false).to_bool()?);
        Ok(())
    }

    #[test]
    fn test_no_vars() -> ExResult<()> {
        fn test_int(s: &str, reference: i32) -> ExResult<()> {
            println!("=== testing\n{}", s);
            let res = parse_val::<i32, f64>(s)?.eval(&[])?.to_int();
            match res {
                Ok(i) => {
                    assert_eq!(reference, i);
                }
                Err(e) => {
                    println!("{:?}", e);
                    unreachable!();
                }
            }
            Ok(())
        }
        fn test_float(s: &str, reference: f64) -> ExResult<()> {
            println!("=== testing\n{}", s);
            let expr = FlatExVal::<i32, f64>::from_str(s)?;
            assert_float_eq_f64(reference, expr.eval(&[])?.to_float()?);
            Ok(())
        }
        fn test_bool(s: &str, reference: bool) -> ExResult<()> {
            println!("=== testing\n{}", s);
            let expr = FlatExVal::<i32, f64>::from_str(s)?;
            assert_eq!(reference, expr.eval(&[])?.to_bool()?);
            Ok(())
        }
        fn test_error(s: &str) -> ExResult<()> {
            let expr = FlatExVal::<i32, f64>::from_str(s);
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
            let expr = FlatExVal::<i32, f64>::from_str(s)?;
            match expr.eval(&[])? {
                Val::None => Ok(()),
                _ => Err(format_exerr!("'{}' should return none but didn't", s)),
            }
        }
        test_int("1+2 if 1 > 0 else 2+4", 3)?;
        test_int("1+2 if 1 < 0 else 2+4", 6)?;
        test_error("929<<92")?;
        test_error("929<<32")?;
        test_error("929>>32")?;
        test_int("928<<31", 0)?;
        test_int("929>>31", 0)?;
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
        test_float("round(π)", 3.0)?;
        test_float("cos(π)", -1.0)?;
        test_float("sin (1 if false else 2.0)", 2.0f64.sin())?;
        test_float("cbrt(27.0)", 3.0)?;
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
        test_float("1 + 1.0", 2.0)?;
        test_bool("1.0 == 1", true)?;
        test_bool("1 == 1", true)?;
        test_bool("2 == true", false)?;
        test_bool("1.5 < 1", false)?;
        test_bool("true == true", true)?;
        test_bool("false != true", true)?;
        test_bool("false != false", false)?;
        test_bool("1 > 0.5", true)?;
        test_error("to_float(10000000000000)")?;
        test_bool("true == 1", false)?;
        test_bool("true else 2", true)?;
        test_int("1 else 2", 1)?;
        test_error("if true else 2")?;
        test_none("2 if false")?;
        test_int("to_int(1)", 1)?;
        test_int("to_int(3.5)", 3)?;
        test_float("to_float(2)", 2.0)?;
        test_float("to_float(3.5)", 3.5)?;
        test_float("to_float(true)", 1.0)?;
        test_float("to_float(false)", 0.0)?;
        test_int("to_int(true)", 1)?;
        test_int("to_int(false)", 0)?;
        test_error("to_int(fact(-1))")?;
        test_error("to_float(5 if false)")?;
        test_error("0/0")?;
        test_bool("(5 if false) == (5 if false)", false)?;
        test_error("2^40")?;
        test_error("1000000000*1000000000")?;
        test_error("1500000000+1500000000")?;
        test_error("-1500000000-1500000000")?;
        test_error("0%0")?;

        Ok(())
    }
}
