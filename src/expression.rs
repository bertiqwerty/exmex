use crate::{
    operators::BinOp,
    util::{apply_unary_ops, CompositionOfUnaryOps},
    ExParseError,
};
use itertools::Itertools;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;

type ExprIdxVec = SmallVec<[usize; 32]>;

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; 32]>;

pub const N_NODES_ON_STACK: usize = 32usize;

pub type FlatNodeVec<T> = SmallVec<[FlatNode<T>; N_NODES_ON_STACK]>;
pub type FlatOpVec<T> = SmallVec<[FlatOp<T>; N_NODES_ON_STACK]>;

/// A `FlatOp` contains besides a binary operation an optional unary operation that
/// will be executed after the binary operation in case of its existence.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatOp<T: Copy> {
    unary_op: Option<CompositionOfUnaryOps<T>>,
    bin_op: BinOp<T>,
}

/// Nodes are inputs for binary operators. A node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Node<T: Copy> {
    Expr(Expression<T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var(usize),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum FlatNodeKind<T: Copy> {
    Num(T),
    Var(usize),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatNode<T: Copy> {
    kind: FlatNodeKind<T>,
    unary_op: Option<CompositionOfUnaryOps<T>>,
}

impl<T: Copy> FlatNode<T> {
    pub fn from_kind(kind: FlatNodeKind<T>) -> FlatNode<T> {
        return FlatNode {
            kind: kind,
            unary_op: None,
        };
    }
}

/// This is the core data type representing a flattened expression and the result of
/// parsing a string. The expression is flattened to make efficient evaluation possible.
/// Simplified, it consists of a [`SmallVec`](SmallVec) of nodes and a
/// [`SmallVec`](SmallVec) of operators that are applied to the nodes in an order following
/// operator priorities.
///
/// You create an expression with the `parse` function or one of its
/// variants, namely `parse_with_default_ops` and `parse_with_number_pattern`.
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{parse_with_default_ops};
///
/// // create an expression by parsing a string
/// let expr = parse_with_default_ops::<f32>("sin(1+y)*x")?;
/// assert!((expr.eval(&[2.0, 1.5])? - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The second argument `&[2.0, 1.5]` in the call of [`eval`](FlatEx::eval) specifies the
/// variable values in the order of their occurrence in the string.
/// In this example, we want to evaluate the expression for the varibale values `y=2.0` and `x=1.5`.
/// Variables in the string to-be-parsed are all substrings that are no numbers, no 
/// operators, and no parentheses.
///
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatEx<T: Copy> {
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
}

fn apply_uop_if_some<T: Copy>(uop: &Option<CompositionOfUnaryOps<T>>, val: T) -> T {
    match uop {
        None => val,
        Some(uops) => apply_unary_ops(&uops, val),
    }
}

impl<T: Copy> FlatEx<T> {
    /// Evaluates an expression with the given variable values and returns the computed
    /// result.
    ///
    /// # Arguments
    ///
    /// * `vars` - Values of the variables of the expression; the n-th value corresponds to
    ///            the n-th variable as given in the string that has been parsed to this expression.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///
    /// # Errors
    ///
    /// If the number of variables in the parsed expression are different from the length of
    /// the variable slice, we return an [`ExParseError`](ExParseError).
    ///
    pub fn eval(&self, vars: &[T]) -> Result<T, ExParseError> {
        if self.n_unique_vars != vars.len() {
            return Err(ExParseError {
                msg: format!(
                    "parsed expression contains {} vars but passed slice has {} elements",
                    self.n_unique_vars,
                    vars.len()
                ),
            });
        }
        let mut numbers = self
            .nodes
            .iter()
            .map(|node| {
                apply_uop_if_some(
                    &node.unary_op,
                    match node.kind {
                        FlatNodeKind::Num(n) => n,
                        FlatNodeKind::Var(idx) => vars[idx],
                    },
                )
            })
            .collect::<SmallVec<[T; 32]>>();
        
        let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> = smallvec![false; N_NODES_ON_STACK];
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = self.prio_indices[i];
            let mut shift_left = 0usize;
            while ignore[num_idx - shift_left] {
                shift_left += 1usize;
            }
            let mut shift_right = 1usize;
            while ignore[num_idx + shift_right] {
                shift_right += 1usize;
            }
            let num_1 = numbers[num_idx - shift_left];
            let num_2 = numbers[num_idx + shift_right];
            numbers[num_idx - shift_left] = {
                let bop_res = (self.ops[bin_op_idx].bin_op.op)(num_1, num_2);
                apply_uop_if_some(&self.ops[bin_op_idx].unary_op, bop_res)
            };
            ignore[num_idx + shift_right] = true;
        }
        Ok(numbers[0])
    }

    fn compile(&mut self) {
        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (FlatNodeKind::Num(num_1), FlatNodeKind::Num(num_2)) =
                (&node_1.kind, &node_2.kind)
            {
                let num_1 = apply_uop_if_some(&node_1.unary_op, *num_1);
                let num_2 = apply_uop_if_some(&node_2.unary_op, *num_2);
                let val = apply_uop_if_some(
                    &self.ops[bin_op_idx].unary_op,
                    (self.ops[bin_op_idx].bin_op.op)(num_1, num_2),
                );
                self.nodes[num_idx] = FlatNode {
                    kind: FlatNodeKind::Num(val),
                    unary_op: None,
                };
                self.nodes.remove(num_idx + 1);
                // reduce indices after removed position
                for num_idx_after in num_inds.iter_mut() {
                    if *num_idx_after > num_idx {
                        *num_idx_after = *num_idx_after - 1;
                    }
                }
                used_prio_indices.push(bin_op_idx);
            } else {
                break;
            }
        }

        self.ops = self
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|x| x.1.clone())
            .collect();

        self.prio_indices = prioritized_indices_flat(&self.ops, &self.nodes);
    }
}

fn flatten_vecs<T: Copy>(expr: &Expression<T>, prio_offset: i32) -> (FlatNodeVec<T>, FlatOpVec<T>) {
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    for (node_idx, node) in expr.nodes.iter().enumerate() {
        match node {
            Node::Num(num) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Num(*num));
                flat_nodes.push(flat_node);
            }
            Node::Var(idx) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Var(*idx));
                flat_nodes.push(flat_node);
            }
            Node::Expr(e) => {
                let (mut sub_nodes, mut sub_ops) = flatten_vecs(e, prio_offset + 100i32);
                flat_nodes.append(&mut sub_nodes);
                flat_ops.append(&mut sub_ops);
            }
        };
        if node_idx < expr.bin_ops.len() {
            let prio_adapted_bin_op = BinOp {
                op: expr.bin_ops[node_idx].op,
                prio: expr.bin_ops[node_idx].prio + prio_offset,
            };
            flat_ops.push(FlatOp {
                bin_op: prio_adapted_bin_op,
                unary_op: None,
            });
        }
    }

    if expr.unary_op.len() > 0 {
        if flat_ops.len() > 0 {
            // find the last binary operator with the lowest priority of this expression,
            // since this will be executed as the last one
            let low_prio_op = match flat_ops.iter_mut().rev().min_by_key(|op| op.bin_op.prio) {
                None => panic!("cannot have more than one flat node but no binary ops"),
                Some(x) => x,
            };
            let mut new_uops = expr.unary_op.clone();
            let mut dummy = CompositionOfUnaryOps::<T>::new();
            let mut existing_uops = match &mut low_prio_op.unary_op {
                Some(uops) => uops,
                None => &mut dummy,
            };
            new_uops.append(&mut existing_uops);
            *low_prio_op = FlatOp {
                bin_op: low_prio_op.bin_op,
                unary_op: Some(new_uops),
            }
        } else {
            let mut new_op = expr.unary_op.clone();
            flat_nodes[0].unary_op = match &mut flat_nodes[0].unary_op {
                None => Some(new_op),
                Some(uops) => {
                    new_op.append(uops);
                    Some(new_op)
                }
            }
        }
    }
    (flat_nodes, flat_ops)
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Expression<T: Copy> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<Node<T>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: BinOpVec<T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators. The last unary operator is applied first to the result
    /// of the evaluation of nodes and binary operators
    unary_op: CompositionOfUnaryOps<T>,
    prio_indices: ExprIdxVec,
}

fn prioritized_indices_flat<T: Copy>(ops: &[FlatOp<T>], nodes: &FlatNodeVec<T>) -> ExprIdxVec {
    let mut indices: ExprIdxVec = (0..ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let (prio_i1, prio_i2) = match (&nodes[*i1].kind, &nodes[*i2].kind) {
            (FlatNodeKind::Num(_), FlatNodeKind::Num(_)) => {
                let prio_inc = 5;
                (
                    &ops[*i1].bin_op.prio * 10 + prio_inc,
                    &ops[*i2].bin_op.prio * 10 + prio_inc,
                )
            }
            _ => (&ops[*i1].bin_op.prio * 10, &ops[*i2].bin_op.prio * 10),
        };
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

fn prioritized_indices<T: Copy>(bin_ops: &[BinOp<T>], nodes: &Vec<Node<T>>) -> ExprIdxVec {
    let mut indices: ExprIdxVec = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let (prio_i1, prio_i2) = match (&nodes[*i1], &nodes[*i2]) {
            (Node::Num(_), Node::Num(_)) => {
                let prio_inc = 5;
                (
                    &bin_ops[*i1].prio * 10 + prio_inc,
                    &bin_ops[*i2].prio * 10 + prio_inc,
                )
            }
            _ => (&bin_ops[*i1].prio * 10, &bin_ops[*i2].prio * 10),
        };
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

impl<T: Copy + Debug> Expression<T> {
    fn compile(&mut self) {
        for node in &mut self.nodes {
            if let Node::Expr(ref mut e) = node {
                e.compile();
            };
        }

        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (Node::Num(num_1), Node::Num(num_2)) = (node_1, node_2) {
                self.nodes[num_idx] = Node::Num((self.bin_ops[bin_op_idx].op)(*num_1, *num_2));
                self.nodes.remove(num_idx + 1);
                // reduce indices after removed position
                for num_idx_after in num_inds.iter_mut() {
                    if *num_idx_after > num_idx {
                        *num_idx_after = *num_idx_after - 1;
                    }
                }
                used_prio_indices.push(bin_op_idx);
            } else {
                break;
            }
        }

        self.bin_ops = self
            .bin_ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|x| *x.1)
            .collect();

        self.prio_indices = prioritized_indices(&self.bin_ops, &self.nodes);
    }
    pub fn new(
        nodes: Vec<Node<T>>,
        bin_ops: BinOpVec<T>,
        unary_op: CompositionOfUnaryOps<T>,
    ) -> Result<Expression<T>, ExParseError> {
        if nodes.len() != bin_ops.len() + 1 {
            Err(ExParseError {
                msg: "mismatch between number of nodes and binary operators".to_string(),
            })
        } else {
            let indices = prioritized_indices(&bin_ops, &nodes);
            let mut expr = Expression {
                nodes: nodes,
                bin_ops: bin_ops,
                unary_op,
                prio_indices: indices,
            };
            expr.compile();
            Ok(expr)
        }
    }
    pub fn flatten(&self) -> FlatEx<T> {
        let (nodes, ops) = flatten_vecs(self, 0);
        let indices = prioritized_indices_flat(&ops, &nodes);
        let n_unique_vars = nodes.iter().filter_map(|n| {
            match n.kind {
                FlatNodeKind::Var(idx) => Some(idx),
                _ => None,
            }            
        }).unique().count();
        let mut flatex = FlatEx {
            nodes: nodes,
            ops: ops,
            prio_indices: indices,
            n_unique_vars: n_unique_vars,
        };
        flatex.compile();
        flatex
    }
}

#[cfg(test)]
mod test {

    use crate::{parse_with_default_ops, util::assert_float_eq_f64};
    #[test]
    fn test_compile() {
        let flat_ex = parse_with_default_ops::<f64>("1*sin(2-0.1)").unwrap();
        assert_float_eq_f64(flat_ex.eval(&[]).unwrap(), 1.9f64.sin());
        assert_eq!(flat_ex.nodes.len(), 1);
    }
}
