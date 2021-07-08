use num::Float;

/// Operators can be custom-defined by the library-user in terms of this struct.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Operator<'a, T: Copy> {
    pub repr: &'a str,
    pub bin_op: Option<BinOp<T>>,
    pub unary_op: Option<fn(T) -> T>,
}

/// A binary operator that consists of a function pointer and a priority.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinOp<T: Copy> {
    pub op: fn(T, T) -> T,
    pub prio: i16,
}

/// Returns the default operators as `VecOps<'a, T>`.
pub fn make_default_operators<'a, T: Float>() -> Vec<Operator<'a, T>> {
    vec![
        Operator {
            repr: "^",
            bin_op: Some(BinOp {
                op: |a: T, b| a.powf(b),
                prio: 2,
            }),
            unary_op: None,
        },
        Operator {
            repr: "*",
            bin_op: Some(BinOp {
                op: |a, b| a * b,
                prio: 1,
            }),
            unary_op: None,
        },
        Operator {
            repr: "/",
            bin_op: Some(BinOp {
                op: |a, b| a / b,
                prio: 1,
            }),
            unary_op: None,
        },
        Operator {
            repr: "+",
            bin_op: Some(BinOp {
                op: |a, b| a + b,
                prio: 0,
            }),
            unary_op: Some(|a: T| a),
        },
        Operator {
            repr: "-",
            bin_op: Some(BinOp {
                op: |a, b| a - b,
                prio: 0,
            }),
            unary_op: Some(|a: T| (-a)),
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
            repr: "exp",
            bin_op: None,
            unary_op: Some(|a: T| a.exp()),
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
