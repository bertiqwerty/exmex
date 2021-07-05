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
    // the last unary operator is applied first to the result
    // of the evaluation of nodes and binary operators
    pub unary_ops: Vec<fn(T) -> T>  
}


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinOp<T: Copy> {
    pub op: fn(T, T) -> T,
    pub prio: i16,
}


