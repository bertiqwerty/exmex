use num::Float;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;

use super::{
    deep::{BinOpsWithReprs, DeepEx, ExprIdxVec},
    deep_details::{self, find_overloaded_ops, OverloadedOps},
};
use crate::{
    definitions::N_BINOPS_OF_DEEPEX_ON_STACK,
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

#[derive(Debug)]
pub struct PartialDerivative<'a, T: Copy + Debug> {
    repr: &'a str,
    bin_op: Option<
        fn(
            i: usize,
            DeepEx<'a, T>,
            DeepEx<'a, T>,
            &'a [Operator<'a, T>],
        ) -> Result<DeepEx<'a, T>, ExParseError>,
    >,
    unary_op: Option<UnaryOpWithReprs<'a, T>>,
}

fn find_as_bin_op_with_reprs<'a, T: Copy + Debug>(
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

fn find_as_unary_op_with_reprs<'a, T: Copy + Debug>(
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

fn make_op_missing_err(repr: &str) -> ExParseError {
    ExParseError {
        msg: format!("operator {} needed for partial derivative", repr),
    }
}

fn partial_derivative_outer<'a, T: Float + Debug>(
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let bin_ops = deepex.bin_ops.clone();
    let nodes = deepex.nodes().clone();
    let factorexes =
        deepex
            .unary_op()
            .reprs
            .iter()
            .map(|repr| -> Result<DeepEx<'a, T>, ExParseError> {
                let op = partial_derivative_ops
                    .iter()
                    .find(|pdo| &pdo.repr == repr)
                    .ok_or(make_op_missing_err(repr))?;
                let unary_op = op.unary_op.clone().ok_or(make_op_missing_err(repr))?;

                DeepEx::new(
                    nodes.clone(),
                    BinOpsWithReprs {
                        ops: bin_ops.ops.clone(),
                        reprs: bin_ops.reprs.clone(),
                    },
                    unary_op,
                )
            });
    let resex = factorexes.fold(
        Ok(DeepEx::from_node(
            DeepNode::Num(T::from(1.0).unwrap()),
            deepex.overloaded_ops().clone().ok_or(ExParseError {
                msg: "need overloaded ops for outer derivatives".to_string(),
            })?,
        )),
        |dp1, dp2| -> Result<DeepEx<T>, ExParseError> { Ok(dp1? * dp2?) },
    );
    resex
}

fn partial_derivative_inner<'a, T: Float + Debug>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
    overloaded_ops: OverloadedOps<'a, T>,
    ops: &'a [Operator<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let mut nodes = deepex.nodes().clone();
    // special case, partial derivative of only 1 node
    if nodes.len() == 1 {
        let node = nodes.pop().unwrap();
        let zero_node = DeepNode::Num(T::from(0.0).unwrap());
        let one_node = DeepNode::Num(T::from(1.0).unwrap());

        match node {
            DeepNode::Num(_) => return Ok(DeepEx::from_node(zero_node, overloaded_ops.clone())),
            DeepNode::Var((var_i, _)) => {
                return if var_i == var_idx {
                    Ok(DeepEx::from_node(one_node, overloaded_ops.clone()))
                } else {
                    Ok(DeepEx::from_node(zero_node, overloaded_ops.clone()))
                };
            }
            DeepNode::Expr(mut e) => {
                e.set_overloaded_ops(Some(overloaded_ops.clone()));
                return partial_deepex(var_idx, e, ops);
            }
        }
    }

    let partial_bin_ops_of_deepex =
        deepex
            .bin_ops()
            .reprs
            .iter()
            .map(|repr| -> Result<&PartialDerivative<'a, T>, ExParseError> {
                partial_derivative_ops
                    .iter()
                    .find(|pdo| &pdo.repr == repr)
                    .ok_or(ExParseError {
                        msg: format!(
                            "derivative operator of {} needed for partial derivative",
                            repr
                        ),
                    })
            })
            .collect::<Result<
                SmallVec<[&PartialDerivative<'a, T>; N_BINOPS_OF_DEEPEX_ON_STACK]>,
                ExParseError,
            >>()?;

    let prio_indices = deep_details::prioritized_indices(&deepex.bin_ops().ops, &nodes);
    let mut num_inds = prio_indices.clone();
    let mut used_prio_indices = ExprIdxVec::new();
    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let node_1 = &nodes[num_idx];
        let node_2 = &nodes[num_idx + 1];
        let pdo = &partial_bin_ops_of_deepex[bin_op_idx];
        let pd_deepex = (pdo.bin_op.unwrap())(
            var_idx,
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
    Ok(res)
}

pub fn partial_deepex<'a, T: Float + Debug + 'a>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    ops: &'a [Operator<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let partial_derivative_ops = make_partial_derivative_ops::<T>();
    let overloaded_ops = find_overloaded_ops(ops).ok_or(ExParseError {
        msg: "one of overloaded ops not found".to_string(),
    })?;
    
    let inner = partial_derivative_inner(
        var_idx,
        deepex.clone(),
        &partial_derivative_ops,
        overloaded_ops.clone(),
        ops,
    )?;
    let outer = partial_derivative_outer(deepex, &partial_derivative_ops)?;
    let mut res = inner * outer;
    res.set_overloaded_ops(Some(overloaded_ops));
    Ok(res)
}

pub fn make_partial_derivative_ops<'a, T: Float + Debug>() -> [PartialDerivative<'a, T>; 4] {
    [
        PartialDerivative {
            repr: "^",
            bin_op: Some(
                |i: usize,
                 f: DeepEx<T>,
                 g: DeepEx<T>,
                 ops: &'a [Operator<'a, T>]|
                 -> Result<DeepEx<T>, ExParseError> {
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let log_op = find_as_unary_op_with_reprs("log", ops)?;

                    let one = match f.overloaded_ops() {
                        Some(ovops) => {
                            DeepEx::from_node(DeepNode::Num(T::from(1.0).unwrap()), ovops.clone())
                        }
                        None => Err(ExParseError {
                            msg: "overloaded operators needed for partial derivatives".to_string(),
                        })?,
                    };

                    Ok(f.clone().operate_bin(g.clone() - one, power_op.clone())
                        * g.clone()
                        * partial_deepex(i, f.clone(), ops)?
                        + f.clone().operate_bin(g.clone(), power_op)
                            * f.operate_unary(log_op)
                            * partial_deepex(i, g, ops)?)
                },
            ),
            unary_op: None,
        },
        PartialDerivative {
            repr: "+",
            bin_op: Some(
                |i: usize,
                 f: DeepEx<T>,
                 g: DeepEx<T>,
                 ops: &'a [Operator<'a, T>]|
                 -> Result<DeepEx<T>, ExParseError> {
                    Ok(partial_deepex(i, f, ops)? + partial_deepex(i, g, ops)?)
                },
            ),
            unary_op: Some(UnaryOpWithReprs::from_tuple(("-sin", |a: T| a))),
        },
        PartialDerivative {
            repr: "sin",
            bin_op: None,
            unary_op: Some(UnaryOpWithReprs::from_tuple(("cos", |a: T| a.cos()))),
        },
        PartialDerivative {
            repr: "cos",
            bin_op: None,
            unary_op: Some(UnaryOpWithReprs::from_tuple(("-sin", |a: T| -a.sin()))),
        },
    ]
}

#[cfg(test)]
use {
    super::flat::flatten,
    crate::{operators::make_default_operators, util::assert_float_eq_f64},
};

#[test]
fn test_partial_derivative_second_var() {
    let ops = make_default_operators::<f64>();
    let deepex = DeepEx::<f64>::from_str("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(1, deepex.clone(), &ops).unwrap();
    let result = flatten(derivative.clone()).eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = flatten(derivative).eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.8414709848078965);
}

#[test]
fn test_partial_derivative_first_var() {
    let ops = make_default_operators::<f64>();

    let deepex = DeepEx::<f64>::from_str("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = flatten(derivative.clone()).eval(&[0.0, 2345.03]).unwrap();
    assert_float_eq_f64(result, 1.0);
    let result = flatten(derivative).eval(&[1.0, 43212.43]).unwrap();
    assert_float_eq_f64(result, 0.5403023058681398);
    
}

#[test]
fn test_partial_outer() {
    let partial_derivative_ops = make_partial_derivative_ops::<f64>();
    let deepex_1 = DeepEx::<f64>::from_str("sin(x)").unwrap();
    let deepex = deepex_1.nodes()[0].clone();

    match deepex {
        DeepNode::Expr(mut e) => {
            e.set_overloaded_ops(deepex_1.overloaded_ops().clone());
            let deri = partial_derivative_outer(e, &partial_derivative_ops).unwrap();
            assert_eq!(deri.nodes().len(), 2);
            let flatex = flatten(deri);
            assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 0.5403023058681398);
            assert_float_eq_f64(flatex.eval(&[0.0]).unwrap(), 1.0);
            assert_float_eq_f64(flatex.eval(&[2.0]).unwrap(), -0.4161468365471424);
        }
        _ => (),
    }
}

#[test]
fn test_partial_derivative() {
    let ops = make_default_operators::<f64>();

    let deepex = DeepEx::<f64>::from_str("1").unwrap();
    let derivative = partial_deepex(0, deepex, &ops).unwrap();

    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 0.0),
        _ => assert!(false),
    }
    let deepex = DeepEx::<f64>::from_str("x").unwrap();
    let derivative = partial_deepex(0, deepex, &ops).unwrap();
    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 1.0),
        _ => assert!(false),
    }
    let deepex = DeepEx::<f64>::from_str("x^2").unwrap();
    let derivative = partial_deepex(0, deepex, &ops).unwrap();
    let result = flatten(derivative).eval(&[4.5]).unwrap();
    assert_float_eq_f64(result, 9.0);

    let deepex = DeepEx::<f64>::from_str("sin(x)").unwrap();

    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = flatten(derivative.clone()).eval(&[0.0]).unwrap();
    assert_float_eq_f64(result, 1.0);
    let result = flatten(derivative).eval(&[1.0]).unwrap();
    assert_float_eq_f64(result, 0.5403023058681398);
}
