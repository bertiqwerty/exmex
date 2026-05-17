use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Rem, Shl, Shr, Sub};

pub trait Num:
    Clone
    + Copy
    + PartialEq
    + PartialOrd
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! impl_num {
    ($($t:ty), *) => {
        $(
        impl Num for $t {
            fn zero() -> Self {
                0 as $t
            }
            fn one() -> Self {
                1 as $t
            }
        }
        )*
    };
}

impl_num!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);

// no argument, same return type
macro_rules! define_fn {
    ($($name:ident), *) => {
        $(
            fn $name(self) -> Self;
        )*
    };
}
// no argument, other return type
macro_rules! define_fn_treturn {
    ($t:ty, $($name:ident), *) => {
        $(
            fn $name(self) -> $t;
        )*
    };
}
// 1 argument same type, same return type
macro_rules! define_fn_1 {
    ($($name:ident), *) => {
        $(
            fn $name(self, n: Self) -> Self;
        )*
    };
}
// 1 argument other type, same return type
macro_rules! define_fn_1_t2 {
    ($other:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: $other) -> Self;
        )*
    };
}
// 1 argument other type, same return type
macro_rules! define_fn_1_t2_checked {
    ($other:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: $other) -> Option<Self>;
        )*
    };
}
pub trait Float: Num + Neg<Output = Self> {
    fn from_f64(f: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_f32(f: f32) -> Self;
    fn from_int<I: PrimInt>(n: I) -> Self;

    define_fn!(
        log2, log10, abs, sin, cos, tan, asin, acos, atan, sqrt, signum, sinh, cosh, tanh, asinh,
        acosh, atanh, floor, ceil, round, trunc, fract, exp, cbrt, ln
    );
    define_fn_treturn!(bool, is_nan);
    define_fn_1!(powf, log, min, max, atan2);
    define_fn_1_t2!(i32, powi);
}

macro_rules! implement_fn {
    ($self:ty, $($name:ident), *) => {
        $(
            fn $name(self) -> Self {
                <$self>::$name(self)
            }
        )*
    };
}
macro_rules! implement_fn_treturn {
    ($self:ty, $t:ty, $($name:ident), *) => {
        $(
            fn $name(self) -> $t {
                <$self>::$name(self)
            }
        )*
    };
}
macro_rules! implement_fn_1 {
    ($self:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: Self) -> Self {
                <$self>::$name(self, n)
            }
        )*
    };
}
macro_rules! implement_fn_1_t2 {
    ($self:ty, $other:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: $other) -> Self {
                <$self>::$name(self, n)
            }
        )*
    };
}
macro_rules! implement_fn_1_t2_checked {
    ($self:ty, $other:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: $other) -> Option<Self> {
                <$self>::$name(self, n)
            }
        )*
    };
}

macro_rules! impl_float {
    ($($t:ty), *) => {
        $(
        impl Float for $t {
            implement_fn!(
                $t, log2, log10, abs, sin, cos, tan, asin, acos, atan, sqrt, signum, sinh, cosh, tanh,
                asinh, acosh, atanh, floor, ceil, round, trunc, fract, exp, cbrt, ln
            );
            implement_fn_treturn!($t, bool, is_nan);
            implement_fn_1!($t, powf, log, min, max, atan2);
            implement_fn_1_t2!($t, i32, powi);
            fn from_f32(f: f32) -> Self {
                f as $t
            }
            fn from_f64(f: f64) -> Self {
                f as $t
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_int<I: PrimInt>(n: I) -> Self {
                n.to_i64() as $t
            }
        }
        )*
    };
}

impl_float!(f32, f64);

pub trait Signed: Num + Neg<Output = Self> {
    fn abs(self) -> Self;
    fn signum(self) -> Self;
}
pub trait PrimInt:
    Num
    + Ord
    + Eq
    + Rem<Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + Shl<Output = Self>
    + Shr<Output = Self>
{
    fn from_u8(n: u8) -> Self;
    fn from_u64(n: u64) -> Self;
    fn to_usize(self) -> usize;
    fn to_u64(self) -> u64;
    fn to_i64(self) -> i64;
    fn to_i32(self) -> i32;
    fn to_u32(self) -> u32;
    fn from_float<F: Float>(f: F) -> Self;
    define_fn!(to_le, to_be, swap_bytes);
    define_fn_treturn!(u32, count_ones, count_zeros);
    define_fn_1_t2_checked!(u32, checked_pow);
    define_fn_1_t2_checked!(Self, checked_add, checked_sub, checked_mul, checked_div);
}

macro_rules! impl_primint {
    ($($t:ty), *) => {
        $(
        impl PrimInt for $t {

            fn from_u8(n: u8) -> Self {
                n as $t
            }
            fn from_u64(n: u64) -> Self {
                n as $t
            }
            fn to_usize(self) -> usize {
                self as usize
            }
            fn to_i64(self) -> i64 {
                self as i64
            }
            fn to_u64(self) -> u64 {
                self as u64
            }
            fn to_i32(self) -> i32 {
                self as i32
            }
            fn to_u32(self) -> u32 {
                self as u32
            }
            fn from_float<F: Float>(f: F) -> Self {
                f.to_f64() as $t
            }
            implement_fn!($t, to_le, to_be, swap_bytes);
            implement_fn_treturn!($t, u32, count_ones, count_zeros);
            implement_fn_1_t2_checked!($t, u32, checked_pow);
            implement_fn_1_t2_checked!($t, $t, checked_add, checked_sub, checked_mul, checked_div);
        }
        )*
    };
}

impl_primint!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! impl_signed {
    ($($t:ty), *) => {
        $(
        impl Signed for $t {
            implement_fn!($t, abs, signum);
        }
        )*
    };
}

impl_signed!(i8, i16, i32, i64, i128, isize);
