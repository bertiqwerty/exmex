use crate::{
    operators::{BinOp, UnaryOp},
    ExParseError,
};
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
    unary_op: UnaryOp<T>,
    bin_op: BinOp<T>,
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum FlatNodeKind<T: Copy> {
    Num(T),
    Var(usize),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatNode<T: Copy> {
    kind: FlatNodeKind<T>,
    unary_op: UnaryOp<T>,
}

impl<T: Copy> FlatNode<T> {
    pub fn from_kind(kind: FlatNodeKind<T>) -> FlatNode<T> {
        return FlatNode {
            kind: kind,
            unary_op: UnaryOp::new(),
        };
    }
}

/// This is the core data type representing a flattened expression and the result of
/// parsing a string. We use flattened expressions to make efficient evaluation possible.
/// Simplified, a flat expression consists of a [`SmallVec`](SmallVec) of nodes and a
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
pub struct FlatEx<'a, T: Copy> {
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
    deep_ex: Option<DeepEx<'a, T>>
}

impl<'a, T: Copy + Debug> FlatEx<'a, T> {
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
                node.unary_op.apply(match node.kind {
                    FlatNodeKind::Num(n) => n,
                    FlatNodeKind::Var(idx) => vars[idx],
                })
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
                let bop_res = (self.ops[bin_op_idx].bin_op.apply)(num_1, num_2);
                self.ops[bin_op_idx].unary_op.apply(bop_res)
            };
            ignore[num_idx + shift_right] = true;
        }
        Ok(numbers[0])
    }
}

fn flatten_vecs<T: Copy>(
    deep_expr: &DeepEx<T>,
    prio_offset: i32,
) -> (FlatNodeVec<T>, FlatOpVec<T>) {
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    for (node_idx, node) in deep_expr.nodes.iter().enumerate() {
        match node {
            DeepNode::Num(num) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Num(*num));
                flat_nodes.push(flat_node);
            }
            DeepNode::Var(idx) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Var(*idx));
                flat_nodes.push(flat_node);
            }
            DeepNode::Expr(e) => {
                let (mut sub_nodes, mut sub_ops) = flatten_vecs(e, prio_offset + 100i32);
                flat_nodes.append(&mut sub_nodes);
                flat_ops.append(&mut sub_ops);
            }
        };
        if node_idx < deep_expr.bin_ops.1.len() {
            let prio_adapted_bin_op = BinOp {
                apply: deep_expr.bin_ops.1[node_idx].apply,
                prio: deep_expr.bin_ops.1[node_idx].prio + prio_offset,
            };
            flat_ops.push(FlatOp {
                bin_op: prio_adapted_bin_op,
                unary_op: UnaryOp::new(),
            });
        }
    }

    if deep_expr.unary_op.1.len() > 0 {
        if flat_ops.len() > 0 {
            // find the last binary operator with the lowest priority of this expression,
            // since this will be executed as the last one
            let low_prio_op = match flat_ops.iter_mut().rev().min_by_key(|op| op.bin_op.prio) {
                None => panic!("cannot have more than one flat node but no binary ops"),
                Some(x) => x,
            };
            low_prio_op
                .unary_op
                .append_front(&mut deep_expr.unary_op.1.clone());
        } else {
            flat_nodes[0]
                .unary_op
                .append_front(&mut deep_expr.unary_op.1.clone());
        }
    }
    (flat_nodes, flat_ops)
}

/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum DeepNode<'a, T: Copy> {
    Expr(DeepEx<'a, T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var(usize),
}

/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T: Copy> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: (Vec<&'a str>, BinOpVec<T>),
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: (Vec<&'a str>, UnaryOp<T>),
    prio_indices: ExprIdxVec,

}

fn prioritized_indices_flat<T: Copy>(ops: &[FlatOp<T>], nodes: &FlatNodeVec<T>) -> ExprIdxVec {
    let prio_increase =
        |bin_op_idx: usize| match (&nodes[bin_op_idx].kind, &nodes[bin_op_idx + 1].kind) {
            (FlatNodeKind::Num(_), FlatNodeKind::Num(_)) => {
                let prio_inc = 5;
                &ops[bin_op_idx].bin_op.prio * 10 + prio_inc
            }
            _ => &ops[bin_op_idx].bin_op.prio * 10,
        };
    let mut indices: ExprIdxVec = (0..ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let prio_i1 = prio_increase(*i1);
        let prio_i2 = prio_increase(*i2);
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

fn prioritized_indices<T: Copy>(bin_ops: &[BinOp<T>], nodes: &[DeepNode<T>]) -> ExprIdxVec {
    let prio_increase = |bin_op_idx: usize| match (&nodes[bin_op_idx], &nodes[bin_op_idx + 1]) {
        (DeepNode::Num(_), DeepNode::Num(_)) => {
            let prio_inc = 5;
            &bin_ops[bin_op_idx].prio * 10 + prio_inc
        }
        _ => &bin_ops[bin_op_idx].prio * 10,
    };

    let mut indices: ExprIdxVec = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let prio_i1 = prio_increase(*i1);
        let prio_i2 = prio_increase(*i2);
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

impl<'a, T: Copy + Debug> DeepEx<'a, T> {
    /// Evaluates all operators with numbers as operands.
    pub fn compile(&mut self) {
        // change from exression to number if an expression contains only a number
        for node in &mut self.nodes {
            if let DeepNode::Expr(ref e) = node {
                if e.nodes.len() == 1 {
                    match e.nodes[0] {
                        DeepNode::Num(n) => {
                            *node = DeepNode::Num(n);
                        }
                        _ => (),
                    }
                }
            };
        }
        // after changing from expressions to numbers where possible the prios might change
        self.prio_indices = prioritized_indices(&self.bin_ops.1, &self.nodes);

        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (DeepNode::Num(num_1), DeepNode::Num(num_2)) = (node_1, node_2) {
                let bin_op_result = (self.bin_ops.1[bin_op_idx].apply)(*num_1, *num_2);
                self.nodes[num_idx] = DeepNode::Num(bin_op_result);
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

        self.bin_ops.1 = self
            .bin_ops.1
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|x| *x.1)
            .collect();

        if self.nodes.len() == 1 {
            match self.nodes[0] {
                DeepNode::Num(n) => {
                    self.nodes[0] = DeepNode::Num(self.unary_op.1.apply(n));
                    self.unary_op.0.clear();
                    self.unary_op.1.clear();
                }
                _ => (),
            }
        }
        self.prio_indices = prioritized_indices(&self.bin_ops.1, &self.nodes);
    }
    pub fn new(
        nodes: Vec<DeepNode<'a, T>>,
        bin_ops: (Vec<&'a str>, BinOpVec<T>),
        unary_op: (Vec<&'a str>, UnaryOp<T>),
    ) -> Result<DeepEx<'a, T>, ExParseError> {
        if nodes.len() != bin_ops.1.len() + 1 {
            Err(ExParseError {
                msg: "mismatch between number of nodes and binary operators".to_string(),
            })
        } else {
            let indices = prioritized_indices(&bin_ops.1, &nodes);
            let mut expr = DeepEx {
                nodes: nodes,
                bin_ops: bin_ops,
                unary_op,
                prio_indices: indices,
            };
            expr.compile();
            Ok(expr)
        }
    }
}

pub fn flatten<T: Copy>(deep_ex: DeepEx<T>) -> FlatEx<T> {
    let (nodes, ops) = flatten_vecs(&deep_ex, 0);
    let indices = prioritized_indices_flat(&ops, &nodes);
    let mut found_vars = SmallVec::<[usize; 16]>::new();
    let n_unique_vars = nodes
        .iter()
        .filter_map(|n| match n.kind {
            FlatNodeKind::Var(idx) => {
                if !found_vars.contains(&idx) {
                    found_vars.push(idx);
                    Some(idx)
                } else {
                    None
                }
            }
            _ => None,
        })
        .count();
    FlatEx {
        nodes: nodes,
        ops: ops,
        prio_indices: indices,
        n_unique_vars: n_unique_vars,
        deep_ex: Some(deep_ex)
    }
}

#[cfg(test)]
use crate::{make_default_operators, parse_with_default_ops, util::assert_float_eq_f64};
#[test]
fn test_flat_compile() {
    let flat_ex = parse_with_default_ops::<f64>("1*sin(2-0.1)").unwrap();
    assert_float_eq_f64(flat_ex.eval(&[]).unwrap(), 1.9f64.sin());
    assert_eq!(flat_ex.nodes.len(), 1);

    let flat_ex = parse_with_default_ops::<f64>("x*(2*(2*(2*4*8)))").unwrap();
    assert_float_eq_f64(flat_ex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flat_ex.nodes.len(), 2);

    let flat_ex = parse_with_default_ops::<f64>("1*sin(2-0.1) + x").unwrap();
    assert_float_eq_f64(flat_ex.eval(&[0.0]).unwrap(), 1.9f64.sin());
    assert_eq!(flat_ex.nodes.len(), 2);
    match flat_ex.nodes[0].kind {
        FlatNodeKind::Num(n) => assert_float_eq_f64(n, 1.9f64.sin()),
        _ => assert!(false),
    }
    match flat_ex.nodes[1].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => assert!(false),
    }

    let flat_ex = parse_with_default_ops::<f64>("y + 1 - cos(1/(1*sin(2-0.1))-2) + 2 + x").unwrap();
    assert_eq!(flat_ex.nodes.len(), 3);
    match flat_ex.nodes[0].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => assert!(false),
    }
    match flat_ex.nodes[1].kind {
        FlatNodeKind::Num(_) => (),
        _ => assert!(false),
    }
    match flat_ex.nodes[2].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 1),
        _ => assert!(false),
    }
}

#[test]
fn test_deep_compile() {
    let ops = make_default_operators();
    let nodes = vec![DeepNode::Num(4.5), DeepNode::Num(0.5), DeepNode::Num(1.4)];
    let bin_ops = (vec![ops[1].repr, ops[3].repr], smallvec![ops[1].bin_op.unwrap(), ops[3].bin_op.unwrap()]);
    let unary_op = (vec![ops[6].repr], UnaryOp::from_vec(smallvec![ops[6].unary_op.unwrap()]));
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();

    let bin_ops = (vec![ops[1].repr, ops[3].repr], smallvec![ops[1].bin_op.unwrap(), ops[3].bin_op.unwrap()]);
    let unary_op = (vec![ops[6].repr], UnaryOp::from_vec(smallvec![ops[6].unary_op.unwrap()]));
    let nodes = vec![
        DeepNode::Num(4.5),
        DeepNode::Num(0.5),
        DeepNode::Expr(deep_ex),
    ];
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();
    assert_eq!(deep_ex.nodes.len(), 1);
    match deep_ex.nodes[0] {
        DeepNode::Num(n) => assert_eq!(deep_ex.unary_op.1.apply(n), n),
        _ => {
            assert!(false);
        }
    }
}
