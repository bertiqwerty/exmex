use crate::{ExError, ExResult, definitions::N_UNARYOPS_OF_DEEPEX_ON_STACK, format_exerr};
use num::Float;
use smallvec::{smallvec, SmallVec};
use std::{fmt::Debug, marker::PhantomData};

enum OperatorType {
    Bin,
    Unary
}

fn make_op_not_available_error(repr: &str, op_type: OperatorType) -> ExError {
    let op_type_str = match op_type {
        OperatorType::Bin => "binary",
        OperatorType::Unary => "unary"
    };
    format_exerr!("{} operator {} not available", op_type_str, repr)    
}

/// Operators can be custom-defined by the library-user in terms of this struct.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Operator<'a, T: Clone> {
    /// Representation of the operator in the string to be parsed, e.g., `-` or `sin`.
    repr: &'a str,
    /// Binary operator that contains a priority besides a function pointer.
    bin_op: Option<BinOp<T>>,
    /// Unary operator that does not have an explicit priority. Unary operators have
    /// higher priority than binary opertors, e.g., `-1^2 == 1`.
    unary_op: Option<fn(T) -> T>,
    /// An operator can also be constant.
    constant: Option<T>,
}

fn unwrap_operator<'a, O>(wrapped_op: &'a Option<O>, repr: &str, op_type: OperatorType) -> ExResult<&'a O> {
    wrapped_op.as_ref().ok_or_else(||make_op_not_available_error(repr, op_type))
}

impl<'a, T: Clone> Operator<'a, T> {
    fn new(
        repr: &'a str,
        bin_op: Option<BinOp<T>>,
        unary_op: Option<fn(T) -> T>,
        constant: Option<T>,
    ) -> Operator<'a, T> {
        if constant.is_some() {
            if bin_op.is_some() {
                panic!(
                    "Bug! Operators cannot be constant and binary. Check '{}'",
                    repr
                );
            }
            if unary_op.is_some() {
                panic!(
                    "Bug! Operators cannot be constant and unary. Check '{}'.",
                    repr
                );
            }
        }
        Operator {
            repr,
            bin_op,
            unary_op,
            constant,
        }
    }

    /// Creates a binary operator.
    pub fn make_bin(repr: &'a str, bin_op: BinOp<T>) -> Operator<'a, T> {
        Operator::new(repr, Some(bin_op), None, None)
    }
    /// Creates a unary operator.
    pub fn make_unary(repr: &'a str, unary_op: fn(T) -> T) -> Operator<'a, T> {
        Operator::new(repr, None, Some(unary_op), None)
    }
    /// Creates an operator that is either unary or binary based on its positioning in the string to be parsed.
    /// For instance, `-` as defined in [`FloatOpsFactory`](FloatOpsFactory) is unary in `-x` and binary
    /// in `2-x`.
    pub fn make_bin_unary(
        repr: &'a str,
        bin_op: BinOp<T>,
        unary_op: fn(T) -> T,
    ) -> Operator<'a, T> {
        Operator::new(repr, Some(bin_op), Some(unary_op), None)
    }
    /// Creates a constant operator. If an operator is constant it cannot be additionally binary or unary.
    pub fn make_constant(repr: &'a str, constant: T) -> Operator<'a, T> {
        Operator::new(repr, None, None, Some(constant))
    }

    pub fn bin(&self) -> ExResult<BinOp<T>> {
        let op = unwrap_operator(&self.bin_op, self.repr, OperatorType::Bin)?;
        Ok(op.clone())
    }
    pub fn unary(&self) -> ExResult<fn(T) -> T> {
        Ok(*unwrap_operator(&self.unary_op, self.repr, OperatorType::Unary)?)
    }
    pub fn repr(&self) -> &'a str {
        self.repr
    }
    pub fn has_bin(&self) -> bool {
        self.bin_op.is_some()
    }
    pub fn has_unary(&self) -> bool {
        self.unary_op.is_some()
    }
    pub fn constant(&self) -> Option<T> {
        self.constant.clone()
    }
}

pub type VecOfUnaryFuncs<T> = SmallVec<[fn(T) -> T; N_UNARYOPS_OF_DEEPEX_ON_STACK]>;

/// Container of unary operators of one expression
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOp<T> {
    funcs_to_be_composed: VecOfUnaryFuncs<T>,
}

impl<T> UnaryOp<T> {
    /// Applies unary operators one after the other starting with the last.
    /// # Arguments
    ///
    /// * `x` - number the unary operators are applied to
    ///
    pub fn apply(&self, x: T) -> T {
        let mut result = x;
        // rev, since the last uop is applied first by convention
        for uo in self.funcs_to_be_composed.iter().rev() {
            result = uo(result);
        }
        result
    }

    pub fn append_front(&mut self, other: &mut UnaryOp<T>) {
        self.funcs_to_be_composed = other
            .funcs_to_be_composed
            .iter()
            .chain(self.funcs_to_be_composed.iter())
            .copied()
            .collect::<SmallVec<_>>();
    }

    pub fn len(&self) -> usize {
        self.funcs_to_be_composed.len()
    }

    pub fn new() -> Self {
        Self {
            funcs_to_be_composed: smallvec![],
        }
    }

    pub fn from_vec(v: VecOfUnaryFuncs<T>) -> Self {
        Self {
            funcs_to_be_composed: v,
        }
    }

    pub fn clear(&mut self) {
        self.funcs_to_be_composed.clear();
    }
}

/// A binary operator that consists of a function pointer, a priority, and a commutativity-flag.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOp<T: Clone> {
    /// Implementation of the binary operation, e.g., `|a, b| a * b` for multiplication.
    pub apply: fn(T, T) -> T,
    /// Priority of the binary operation. A binary operation with a
    /// higher number will be executed first. For instance, in a sane world `*`
    /// has a higher priority than `+`. However, in Exmex land you could also define
    /// this differently.
    pub prio: i32,
    /// True if this is a commutative operator such as `*` or `+`, false if not such as `-`, `/`, or `^`.
    /// Commutativity is used to compile sub-expressions of numbers correctly.
    pub is_commutative: bool,
}

/// To use custom operators one needs to create a factory that implements this trait.
/// In this way, we make sure that we can deserialize expressions with
/// [`serde`](docs.rs/serde) with the correct operators based on the type.
///
/// # Example
///
/// ```rust
/// use exmex::{BinOp, MakeOperators, Operator};
/// #[derive(Clone)]
/// struct SomeOpsFactory;
/// impl MakeOperators<f32> for SomeOpsFactory {
///     fn make<'a>() -> Vec<Operator<'a, f32>> {    
///         vec![
///             Operator::make_bin_unary(
///                 "-",
///                 BinOp {
///                     apply: |a, b| a - b,
///                     prio: 0,
///                     is_commutative: false,
///                 },
///                 |a| (-a),
///             ),
///             Operator::make_unary("sin", |a| a.sin())
///         ]
///     }
/// }
/// ```
pub trait MakeOperators<T: Clone>: Clone {
    /// Function that creates a vector of operators.
    fn make<'a>() -> Vec<Operator<'a, T>>;
}

/// Factory of default operators for floating point values.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FloatOpsFactory<T: Float> {
    dummy: PhantomData<T>,
}

impl<T: Float> MakeOperators<T> for FloatOpsFactory<T> {
    /// Returns the default operators.
    fn make<'a>() -> Vec<Operator<'a, T>> {
        vec![
            Operator::make_bin(
                "^",
                BinOp {
                    apply: |a, b| a.powf(b),
                    prio: 4,
                    is_commutative: false,
                },
            ),
            Operator::make_bin(
                "*",
                BinOp {
                    apply: |a, b| a * b,
                    prio: 2,
                    is_commutative: true,
                },
            ),
            Operator::make_bin(
                "/",
                BinOp {
                    apply: |a, b| a / b,
                    prio: 3,
                    is_commutative: false,
                },
            ),
            Operator::make_bin_unary(
                "+",
                BinOp {
                    apply: |a, b| a + b,
                    prio: 0,
                    is_commutative: true,
                },
                |a| a,
            ),
            Operator::make_bin_unary(
                "-",
                BinOp {
                    apply: |a, b| a - b,
                    prio: 1,
                    is_commutative: false,
                },
                |a| -a,
            ),
            Operator::make_unary("abs", |a| a.abs()),
            Operator::make_unary("signum", |a| a.signum()),
            Operator::make_unary("sin", |a| a.sin()),
            Operator::make_unary("cos", |a| a.cos()),
            Operator::make_unary("tan", |a| a.tan()),
            Operator::make_unary("asin", |a| a.asin()),
            Operator::make_unary("acos", |a| a.acos()),
            Operator::make_unary("atan", |a| a.atan()),
            Operator::make_unary("sinh", |a| a.sinh()),
            Operator::make_unary("cosh", |a| a.cosh()),
            Operator::make_unary("tanh", |a| a.tanh()),
            Operator::make_unary("floor", |a| a.floor()),
            Operator::make_unary("ceil", |a| a.ceil()),
            Operator::make_unary("trunc", |a| a.trunc()),
            Operator::make_unary("fract", |a| a.fract()),
            Operator::make_unary("exp", |a| a.exp()),
            Operator::make_unary("sqrt", |a| a.sqrt()),
            Operator::make_unary("log", |a| a.ln()),
            Operator::make_unary("log2", |a| a.log2()),
            Operator::make_constant("PI", T::from(std::f64::consts::PI).unwrap()),
            Operator::make_constant("π", T::from(std::f64::consts::PI).unwrap()),
            Operator::make_constant("E", T::from(std::f64::consts::E).unwrap()),
        ]
    }
}

/// This macro creates an operator factory struct that implements the trait
/// [`MakeOperators`](MakeOperators). You have to pass the name of the struct
/// as first, the type of the operands as second, and the [`Operator`](Operator)s as
/// third to n-th argument.
///
/// # Example
///
/// The following snippet creates a struct that can be used as in [`FlatEx<_, MyOpsFactory>`](crate::FlatEx).
/// ```
/// use exmex::{MakeOperators, Operator, ops_factory};
/// ops_factory!(
///     MyOpsFactory,  // name of struct
///     f32,           // data type of operands
///     Operator::make_unary("log", |a| a.ln()),
///     Operator::make_unary("log2", |a| a.log2())
/// );
/// ```
#[macro_export]
macro_rules! ops_factory {
    ($name:ident, $T:ty, $( $ops:expr ),*) => {
        #[derive(Clone)]
        struct $name;
        impl MakeOperators<$T> for $name {
            fn make<'a>() -> Vec<Operator<'a, $T>> {
                vec![$($ops,)*]
            }
        }
    }
}
