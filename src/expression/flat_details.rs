use std::fmt::Debug;

use smallvec::{smallvec, SmallVec};

use crate::{
    definitions::N_NODES_ON_STACK,
    operators::{BinOp, UnaryOp},
    ExError, ExResult,
};

use super::flat::ExprIdxVec;
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


pub fn prioritized_indices_flat<T: Clone + Debug>(
    ops: &[FlatOp<T>],
    nodes: &FlatNodeVec<T>,
) -> ExprIdxVec {
    let prio_increase =
        |bin_op_idx: usize| match (&nodes[bin_op_idx].kind, &nodes[bin_op_idx + 1].kind) {
            (FlatNodeKind::Num(_), FlatNodeKind::Num(_)) if ops[bin_op_idx].bin_op.is_commutative => {
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
