use std::fmt::Debug;

use num::Float;
use smallvec::{smallvec, SmallVec};

use super::{
    deep::{BinOpsWithReprs, DeepEx, ExprIdxVec},
    deep_details::{self, find_overloaded_ops},
};
use crate::{
    definitions::{N_BINOPS_OF_DEEPEX_ON_STACK, N_NODES_ON_STACK},
    expression::deep::{DeepNode, UnaryOpWithReprs},
    operators::{Operator, UnaryOp},
    ExParseError,
};

pub fn find_op<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Option<Operator<'a, T>> {
    let found = ops.iter().cloned().find(|op| op.repr == repr);
    match found {
        Some(op) => Some(Operator {
            bin_op: op.bin_op,
            unary_op: op.unary_op,
            repr: repr,
        }),
        None => None,
    }
}

pub struct PartialDerivative<'a, T: Copy + Debug> {
    repr: &'a str,
    bin_op: Option<
        fn(
            di: usize,
            DeepEx<'a, T>,
            DeepEx<'a, T>,
            &'a [Operator<'a, T>],
        ) -> Result<DeepEx<'a, T>, ExParseError>,
    >,
    unary_op: Option<fn(DeepEx<'a, T>) -> DeepEx<'a, T>>,
}

fn find_and_convert_bin_op<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Result<BinOpsWithReprs<'a, T>, ExParseError> {
    let op = find_op(repr, ops).ok_or(ExParseError {
        msg: format!("did not find operator {}", repr),
    })?;
    Ok(BinOpsWithReprs {
        reprs: vec![op.repr],
        ops: smallvec![op.bin_op.ok_or(ExParseError {
            msg: format!("operater {} is not binary", op.repr)
        })?],
    })
}

fn find_and_convert_unary_op<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Result<UnaryOpWithReprs<'a, T>, ExParseError> {
    let op = find_op(repr, ops).ok_or(ExParseError {
        msg: format!("did not find operator {}", repr),
    })?;
    Ok(UnaryOpWithReprs {
        reprs: vec![op.repr],
        op: UnaryOp::from_vec(smallvec![op.unary_op.ok_or(ExParseError {
            msg: format!("operater {} is not unary", op.repr)
        })?]),
    })
}

fn find_derivative_ops<'a, T: Copy + Debug>(
    repr: &'a str,
    partial_derivative_ops: &'a [PartialDerivative<'a, T>],
) -> Option<&'a PartialDerivative<'a, T>> {
    Some(partial_derivative_ops.iter().find(|pdo| pdo.repr == repr)?)
}

pub fn partial_deepex<'a, T: Float + Debug + 'a>(
    i: usize,
    deepex: DeepEx<'a, T>,
    ops: &'a [Operator<'a, T>],
) -> DeepEx<'a, T> {
    let partial_derivative_ops = make_partial_derivative_ops::<T>();
    let mut nodes = deepex.nodes().clone();
    let overloaded_ops = find_overloaded_ops(ops).unwrap();
    if nodes.len() == 1 {
        let node = nodes.pop().unwrap();
        let zero_node = DeepNode::Num(T::from(0.0).unwrap());
        let one_node = DeepNode::Num(T::from(1.0).unwrap());
        match node {
            DeepNode::Num(_) => return DeepEx::from_node(zero_node, overloaded_ops.clone()),
            DeepNode::Var((var_i, _)) => {
                return if var_i == i {
                    DeepEx::from_node(one_node, overloaded_ops.clone())
                } else {
                    DeepEx::from_node(zero_node, overloaded_ops.clone())
                };
            }
            DeepNode::Expr(e) => {
                return partial_deepex(i, e, ops);
            }
        }
    }
    let partial_ops_of_deepex = deepex
        .bin_ops()
        .reprs
        .iter()
        .map(|repr| {
            partial_derivative_ops
                .iter()
                .find(|pdo| &pdo.repr == repr)
                .unwrap()
        })
        .collect::<SmallVec<[&PartialDerivative<'a, T>; N_BINOPS_OF_DEEPEX_ON_STACK]>>();
    let prio_indices = deep_details::prioritized_indices(&deepex.bin_ops().ops, &nodes);
    let mut num_inds = prio_indices.clone();
    let mut used_prio_indices = ExprIdxVec::new();
    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let node_1 = &nodes[num_idx];
        let node_2 = &nodes[num_idx + 1];
        let pdo = &partial_ops_of_deepex[bin_op_idx];
        let pd_deepex = (pdo.bin_op.unwrap())(
            i,
            DeepEx::from_node(node_1.clone(), overloaded_ops.clone()),
            DeepEx::from_node(node_2.clone(), overloaded_ops.clone()),
            ops,
        )
        .unwrap();

        nodes[num_idx] = DeepNode::Expr(pd_deepex);
        nodes.remove(num_idx + 1);
        // reduce indices after removed position
        for num_idx_after in num_inds.iter_mut() {
            if *num_idx_after > num_idx {
                *num_idx_after = *num_idx_after - 1;
            }
        }
        used_prio_indices.push(bin_op_idx);
    }
    let mut res = DeepEx::from_node(nodes[0].clone(), overloaded_ops.clone());
    res.set_overloaded_ops(deepex.overloaded_ops().clone());
    res
}

pub fn make_partial_derivative_ops<'a, T: Float + Debug>() -> [PartialDerivative<'a, T>; 1] {
    [PartialDerivative {
        repr: "^",
        bin_op: Some(
            |i: usize, f: DeepEx<T>, g: DeepEx<T>, ops: &'a [Operator<'a, T>]| {
                let power_op = find_and_convert_bin_op("^", ops)?;
                let log_op = find_and_convert_unary_op("log", ops)?;
                let mut one = DeepEx::new(
                    vec![DeepNode::Num(T::from(1.0).unwrap())],
                    BinOpsWithReprs::new(),
                    UnaryOpWithReprs::new(),
                )?;
                one.set_overloaded_ops(f.overloaded_ops().clone());

                Ok(f.clone().operate_bin(g.clone() - one, power_op.clone())
                    * g.clone()
                    * partial_deepex(i, f.clone(), ops)
                    + f.clone().operate_bin(g.clone(), power_op)
                        * f.operate_unary(log_op)
                        * partial_deepex(i, g, ops))
            },
        ),
        unary_op: None,
    }]
}

#[cfg(test)]
use {
    super::flat::flatten,
    crate::{operators::make_default_operators, util::assert_float_eq_f64},
};

#[test]
fn test_partial_derivative() {
    let ops = make_default_operators::<f64>();
    let deepex = DeepEx::<f64>::from_str("1").unwrap();
    let derivative = partial_deepex(0, deepex, &ops);

    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 0.0),
        _ => assert!(false),
    }
    let deepex = DeepEx::<f64>::from_str("x").unwrap();
    let derivative = partial_deepex(0, deepex, &ops);
    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 1.0),
        _ => assert!(false),
    }
    let deepex = DeepEx::<f64>::from_str("x^2").unwrap();
    let derivative = partial_deepex(0, deepex, &ops);
    let result = flatten(derivative).eval(&[4.5]).unwrap();
    assert_float_eq_f64(result, 9.0);
}
