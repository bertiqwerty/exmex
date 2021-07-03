use num::Float;

#[derive(Debug)]
pub enum Node<T: Float> {
    Expr(Expression<T>),
    Num(T),
}
#[derive(Debug)]
pub struct Expression<T: Float> {
    pub nodes: Vec<Node<T>>,
    pub bin_ops: Vec<BinaryOperator<T>>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinaryOperator<T: Copy> {
    pub f: fn(T, T) -> T,
    pub priority: i16,
}
