use num::Float;

pub enum Node<'a, T: Float> {
    EXP(&'a Expression<'a, T>),
    NUM(T),
}

pub struct Expression<'a, T: Float> {
    pub nodes: Vec<Node<'a, T>>,
    pub bin_ops: Vec<BinaryOperator<T>>,
}

#[derive(Clone, Copy)]
pub struct BinaryOperator<T: Copy> {
    pub f: fn(T, T) -> T,
    pub priority: i16,
}
