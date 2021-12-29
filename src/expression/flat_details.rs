use std::fmt::Debug;

use smallvec::{smallvec, SmallVec};

use crate::{
    definitions::N_NODES_ON_STACK,
    operators::{BinOp, UnaryOp},
    ExError, ExResult,
};

use super::deep::{DeepEx, DeepNode, ExprIdxVec};

pub type FlatNodeVec<T> = SmallVec<[FlatNode<T>; N_NODES_ON_STACK]>;
pub type FlatOpVec<T> = SmallVec<[FlatOp<T>; N_NODES_ON_STACK]>;

/// A `FlatOp` contains besides a binary operation an optional unary operation that
/// will be executed after the binary operation in case of its existence.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatOp<T: Clone> {
    pub unary_op: UnaryOp<T>,
    pub bin_op: BinOp<T>,
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum FlatNodeKind<T> {
    Num(T),
    Var(usize),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatNode<T> {
    pub kind: FlatNodeKind<T>,
    pub unary_op: UnaryOp<T>,
}

impl<T> FlatNode<T>
where
    T: Clone,
{
    pub fn from_kind(kind: FlatNodeKind<T>) -> FlatNode<T> {
        FlatNode {
            kind,
            unary_op: UnaryOp::new(),
        }
    }
}

pub fn check_partial_index(var_idx: usize, n_vars: usize, unparsed: &str) -> ExResult<()> {
    if var_idx >= n_vars {
        Err(ExError {
            msg: format!(
                "index {} is invalid since we have only {} vars in {}",
                var_idx, n_vars, unparsed
            ),
        })
    } else {
        Ok(())
    }
}

pub fn flatten_vecs<T: Clone + Debug>(
    deep_expr: &DeepEx<T>,
    prio_offset: i64,
) -> (FlatNodeVec<T>, FlatOpVec<T>) {
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    for (node_idx, node) in deep_expr.nodes().iter().enumerate() {
        match node {
            DeepNode::Num(num) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Num(num.clone()));
                flat_nodes.push(flat_node);
            }
            DeepNode::Var((idx, _)) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Var(*idx));
                flat_nodes.push(flat_node);
            }
            DeepNode::Expr(e) => {
                let (mut sub_nodes, mut sub_ops) = flatten_vecs(e, prio_offset + 100i64);
                flat_nodes.append(&mut sub_nodes);
                flat_ops.append(&mut sub_ops);
            }
        };
        if node_idx < deep_expr.bin_ops().ops.len() {
            let prio_adapted_bin_op = BinOp {
                apply: deep_expr.bin_ops().ops[node_idx].apply,
                prio: deep_expr.bin_ops().ops[node_idx].prio + prio_offset,
                is_commutative: deep_expr.bin_ops().ops[node_idx].is_commutative,
            };
            flat_ops.push(FlatOp {
                bin_op: prio_adapted_bin_op,
                unary_op: UnaryOp::new(),
            });
        }
    }

    if deep_expr.unary_op().op.len() > 0 {
        if !flat_ops.is_empty() {
            // find the last binary operator with the lowest priority of this expression,
            // since this will be executed as the last one
            let low_prio_op = match flat_ops.iter_mut().rev().min_by_key(|op| op.bin_op.prio) {
                None => panic!("cannot have more than one flat node but no binary ops"),
                Some(x) => x,
            };
            low_prio_op
                .unary_op
                .append_latest(&mut deep_expr.unary_op().op.clone());
        } else {
            flat_nodes[0]
                .unary_op
                .append_latest(&mut deep_expr.unary_op().op.clone());
        }
    }
    (flat_nodes, flat_ops)
}

pub fn prioritized_indices_flat<T: Clone + Debug>(
    ops: &[FlatOp<T>],
    nodes: &FlatNodeVec<T>,
) -> ExprIdxVec {
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

pub fn eval_flatex<T: Clone + Debug>(
    vars: &[T],
    nodes: &FlatNodeVec<T>,
    ops: &FlatOpVec<T>,
    prio_indices: &ExprIdxVec,
    n_unique_vars: usize,
) -> ExResult<T> {
    if n_unique_vars != vars.len() {
        return Err(ExError {
            msg: format!(
                "parsed expression contains {} vars but passed slice has {} elements",
                n_unique_vars,
                vars.len()
            ),
        });
    }
    let mut numbers = nodes
        .iter()
        .map(|node| {
            node.unary_op.apply(match &node.kind {
                FlatNodeKind::Num(n) => n.clone(),
                FlatNodeKind::Var(idx) => vars[*idx].clone(),
            })
        })
        .collect::<SmallVec<[T; N_NODES_ON_STACK]>>();
    let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> = smallvec![false; nodes.len()];
    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = prio_indices[i];
        let mut shift_left = 0usize;
        while ignore[num_idx - shift_left] {
            shift_left += 1usize;
        }
        let mut shift_right = 1usize;
        while ignore[num_idx + shift_right] {
            shift_right += 1usize;
        }
        let num_1 = numbers[num_idx - shift_left].clone();
        let num_2 = numbers[num_idx + shift_right].clone();
        numbers[num_idx - shift_left] = {
            let bop_res = (ops[bin_op_idx].bin_op.apply)(num_1, num_2);
            ops[bin_op_idx].unary_op.apply(bop_res)
        };
        ignore[num_idx + shift_right] = true;
    }
    Ok(numbers[0].clone())
}
