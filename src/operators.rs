use num::Float;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct OperatorPair<T: Copy> {
    pub bin_op: Option<BinOp<T>>,
    pub unary_op: Option<fn(T) -> T>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinOp<T: Copy> {
    pub op: fn(T, T) -> T,
    pub prio: i16,
}

pub type VecOps<'a, T> = Vec<(&'a str, OperatorPair<T>)>;


pub fn make_default_operators<'a, T: Float>() -> VecOps<'a, T> {
    vec![
        (
            "^",
            OperatorPair {
                bin_op: Some(BinOp {
                    op: |a: T, b| a.powf(b),
                    prio: 2,
                }),
                unary_op: None,
            },
        ),
        (
            "*",
            OperatorPair {
                bin_op: Some(BinOp {
                    op: |a, b| a * b,
                    prio: 1,
                }),
                unary_op: None,
            },
        ),
        (
            "/",
            OperatorPair {
                bin_op: Some(BinOp {
                    op: |a, b| a / b,
                    prio: 1,
                }),
                unary_op: None,
            },
        ),
        (
            "+",
            OperatorPair {
                bin_op: Some(BinOp {
                    op: |a, b| a + b,
                    prio: 0,
                }),
                unary_op: Some(|a: T| a),
            },
        ),
        (
            "-",
            OperatorPair {
                bin_op: Some(BinOp {
                    op: |a, b| a - b,
                    prio: 0,
                }),
                unary_op: Some(|a: T| (-a)),
            },
        ),
        (
            "sin",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.sin()),
            },
        ),
        (
            "cos",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.cos()),
            },
        ),
        (
            "tan",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.tan()),
            },
        ),
        (
            "exp",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.exp()),
            },
        ),
        (
            "log",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.ln()),
            },
        ),
        (
            "log2",
            OperatorPair {
                bin_op: None,
                unary_op: Some(|a: T| a.log2()),
            },
        ),
    ]
}
