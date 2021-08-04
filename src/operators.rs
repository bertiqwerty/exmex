use num::Float;
use smallvec::{smallvec, SmallVec};

/// Operators can be custom-defined by the library-user in terms of this struct.
///
/// # Examples
///
/// ```
/// use exmex::{BinOp, Operator};
/// let ops = vec![
///     Operator {
///         repr: "-",
///         bin_op: Some(BinOp {
///             apply: |a, b| a - b,
///             prio: 0,
///         }),
///         unary_op: Some(|a: f32| (-a)),
///     },
///     Operator {
///         repr: "sin",
///         bin_op: None,
///         unary_op: Some(|a: f32| a.sin()),
///     }
/// ];
/// ```
///
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Operator<'a, T> {
    /// Representation of the operator in the string to be parsed, e.g., `-` or `sin`.
    pub repr: &'a str,
    /// Binary operator that contains a priority besides a function pointer, if available.
    pub bin_op: Option<BinOp<T>>,
    /// Unary operator that does not have an explicit priority. Unary operators have
    /// higher priority than binary opertors, e.g., `-1^2 == 1`.
    pub unary_op: Option<fn(T) -> T>,
}

pub type VecOfUnaryFuncs<T> = SmallVec<[fn(T) -> T; 8]>;

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
            .map(|f| *f)
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

/// A binary operator that consists of a function pointer and a priority.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOp<T> {
    /// Implementation of the binary operation, e.g., `|a, b| a * b` for multiplication.
    pub apply: fn(T, T) -> T,
    /// Priority of the binary operation. A binary operation with a
    /// higher number will be executed first. For instance, in a sane world `*`
    /// has a higher priority than `+`. However, in Exmex land you could also define
    /// this differently.
    pub prio: i32,
}

/// Returns the default operators.
pub fn make_default_operators<'a, T: Float>() -> [Operator<'a, T>; 23] {
    [
        Operator {
            repr: "^",
            bin_op: Some(BinOp {
                apply: |a: T, b| a.powf(b),
                prio: 2,
            }),
            unary_op: None,
        },
        Operator {
            repr: "*",
            bin_op: Some(BinOp {
                apply: |a, b| a * b,
                prio: 1,
            }),
            unary_op: None,
        },
        Operator {
            repr: "/",
            bin_op: Some(BinOp {
                apply: |a, b| a / b,
                prio: 1,
            }),
            unary_op: None,
        },
        Operator {
            repr: "+",
            bin_op: Some(BinOp {
                apply: |a, b| a + b,
                prio: 0,
            }),
            unary_op: Some(|a: T| a),
        },
        Operator {
            repr: "-",
            bin_op: Some(BinOp {
                apply: |a, b| a - b,
                prio: 0,
            }),
            unary_op: Some(|a: T| (-a)),
        },
        Operator {
            repr: "signum",
            bin_op: None,
            unary_op: Some(|a: T| a.signum()),
        },
        Operator {
            repr: "sin",
            bin_op: None,
            unary_op: Some(|a: T| a.sin()),
        },
        Operator {
            repr: "cos",
            bin_op: None,
            unary_op: Some(|a: T| a.cos()),
        },
        Operator {
            repr: "tan",
            bin_op: None,
            unary_op: Some(|a: T| a.tan()),
        },
        Operator {
            repr: "asin",
            bin_op: None,
            unary_op: Some(|a: T| a.asin()),
        },
        Operator {
            repr: "acos",
            bin_op: None,
            unary_op: Some(|a: T| a.acos()),
        },
        Operator {
            repr: "atan",
            bin_op: None,
            unary_op: Some(|a: T| a.atan()),
        },
        Operator {
            repr: "sinh",
            bin_op: None,
            unary_op: Some(|a: T| a.sinh()),
        },
        Operator {
            repr: "cosh",
            bin_op: None,
            unary_op: Some(|a: T| a.cosh()),
        },
        Operator {
            repr: "tanh",
            bin_op: None,
            unary_op: Some(|a: T| a.tanh()),
        },
        Operator {
            repr: "floor",
            bin_op: None,
            unary_op: Some(|a: T| a.floor()),
        },
        Operator {
            repr: "ceil",
            bin_op: None,
            unary_op: Some(|a: T| a.ceil()),
        },
        Operator {
            repr: "trunc",
            bin_op: None,
            unary_op: Some(|a: T| a.trunc()),
        },
        Operator {
            repr: "fract",
            bin_op: None,
            unary_op: Some(|a: T| a.fract()),
        },
        Operator {
            repr: "exp",
            bin_op: None,
            unary_op: Some(|a: T| a.exp()),
        },
        Operator {
            repr: "sqrt",
            bin_op: None,
            unary_op: Some(|a: T| a.sqrt()),
        },
        Operator {
            repr: "log",
            bin_op: None,
            unary_op: Some(|a: T| a.ln()),
        },
        Operator {
            repr: "log2",
            bin_op: None,
            unary_op: Some(|a: T| a.log2()),
        },
    ]
}
