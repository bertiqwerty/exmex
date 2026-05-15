use std::ops::{Add, Div, Mul, Neg, Sub};

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

macro_rules! define_fn {
    ($($name:ident), *) => {
        $(
            fn $name(self) -> Self;
        )*
    };
}
macro_rules! define_fn_1 {
    ($($name:ident), *) => {
        $(
            fn $name(self, n: Self) -> Self;
        )*
    };
}
macro_rules! define_fn_1_t2 {
    ($other:ty, $($name:ident), *) => {
        $(
            fn $name(self, n: $other) -> Self;
        )*
    };
}
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
                n.to_u64() as $t
            }

        }
        )*
    };
}

impl_float!(f32, f64);

pub trait Signed: Num + Neg<Output = Self> {}
pub trait PrimInt: Num + Ord + Eq {
    fn from_u8(n: u8) -> Self;
    fn to_u64(self) -> u64;
    fn to_i32(self) -> i32;
    fn to_u32(self) -> u32;
    fn from_float<F: Float>(f: F) -> Self;
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
            implement_fn_1_t2_checked!($t, u32, checked_pow);
            implement_fn_1_t2_checked!($t, $t, checked_add, checked_sub, checked_mul, checked_div);
        }
        )*
    };
}

impl_primint!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

macro_rules! impl_trait {
    ($trait_name:ident, $($t:ty),*) => {
        $(
            impl $trait_name for $t {}
        )*
    };
}

impl_trait!(Signed, i8, i16, i32, i64, i128, isize);
