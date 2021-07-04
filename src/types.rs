use num::Float;

#[derive(Debug)]
pub enum Node<T: Float> {
    Expr(Expression<T>),
    Num(T),
}
#[derive(Debug)]
pub struct Expression<T: Float> {
    pub nodes: Vec<Node<T>>,
    pub bin_ops: Vec<BinOp<T>>,
    pub unary_op: Option<fn(T) -> T>
}


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinOp<T: Copy> {
    pub op: fn(T, T) -> T,
    pub prio: i16,
}


