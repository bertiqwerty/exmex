use crate::{
    definitions::N_BINOPS_OF_DEEPEX_ON_STACK,
    expression::{
        deep::{BinOpsWithReprs, DeepEx, DeepNode, ExprIdxVec, UnaryOpWithReprs},
        deep_details,
    },
    operators::{Operator, UnaryOp},
    ExError, ExResult,
};
use num::Float;
use smallvec::{smallvec, SmallVec};
use std::fmt::Debug;

#[derive(Clone)]
struct ValueDerivative<'a, T: Copy + Debug> {
    val: DeepEx<'a, T>,
    der: DeepEx<'a, T>,
}

pub fn find_op<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Option<Operator<'a, T>> {
    let found = ops.iter().cloned().find(|op| op.repr == repr);
    found.map(|op| Operator {
        bin_op: op.bin_op,
        unary_op: op.unary_op,
        repr,
    })
}

pub struct PartialDerivative<'a, T: Copy + Debug> {
    repr: &'a str,
    bin_op: Option<
        fn(
            ValueDerivative<'a, T>,
            ValueDerivative<'a, T>,
            &[Operator<'a, T>],
        ) -> ExResult<ValueDerivative<'a, T>>,
    >,
    unary_outer_op: Option<fn(DeepEx<'a, T>, &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>>,
}

fn find_as_bin_op_with_reprs<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> ExResult<BinOpsWithReprs<'a, T>> {
    let op = find_op(repr, ops).ok_or(ExError {
        msg: format!("did not find operator {}", repr),
    })?;
    Ok(BinOpsWithReprs {
        reprs: vec![op.repr],
        ops: smallvec![op.bin_op.ok_or(ExError {
            msg: format!("operater {} is not binary", op.repr)
        })?],
    })
}

fn find_as_unary_op_with_reprs<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> ExResult<UnaryOpWithReprs<'a, T>> {
    let op = find_op(repr, ops).ok_or(ExError {
        msg: format!("did not find operator {}", repr),
    })?;
    Ok(UnaryOpWithReprs {
        reprs: vec![op.repr],
        op: UnaryOp::from_vec(smallvec![op.unary_op.ok_or(ExError {
            msg: format!("operater {} is not unary", op.repr)
        })?]),
    })
}

fn make_op_missing_err(repr: &str) -> ExError {
    ExError {
        msg: format!("operator {} needed for outer partial derivative", repr),
    }
}

fn partial_derivative_outer<'a, T: Float + Debug>(
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
    ops: &[Operator<'a, T>],
) -> ExResult<DeepEx<'a, T>> {
    let factorexes = deepex
        .unary_op()
        .reprs
        .iter()
        .map(|repr| -> ExResult<DeepEx<'a, T>> {
            let op = partial_derivative_ops
                .iter()
                .find(|pdo| &pdo.repr == repr)
                .ok_or_else(|| make_op_missing_err(repr))?;
            let unary_deri_op = op.unary_outer_op.ok_or_else(|| make_op_missing_err(repr))?;
            unary_deri_op(deepex.clone(), ops)
        });
    let mul_op = find_as_bin_op_with_reprs("*", ops)?;
    let resex = factorexes.fold(Ok(DeepEx::one()), |dp1, dp2| -> ExResult<DeepEx<T>> {
        mul(dp1?, dp2?, mul_op.clone())
    });
    resex
}

fn partial_derivative_inner<'a, T: Float + Debug>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
    ops: &[Operator<'a, T>],
) -> ExResult<DeepEx<'a, T>> {
    // special case, partial derivative of only 1 node
    if deepex.nodes().len() == 1 {
        let res = match deepex.nodes()[0].clone() {
            DeepNode::Num(_) => DeepEx::zero(),
            DeepNode::Var((var_i, _)) => {
                if var_i == var_idx {
                    DeepEx::one()
                } else {
                    DeepEx::zero()
                }
            }
            DeepNode::Expr(e) => partial_deepex(var_idx, e, ops)?,
        };
        let (res, _) = res.var_names_union(deepex);
        return Ok(res);
    }

    let prio_indices = deep_details::prioritized_indices(&deepex.bin_ops().ops, deepex.nodes());

    let make_deepex = |node: DeepNode<'a, T>| match node {
        DeepNode::Expr(e) => e,
        _ => DeepEx::from_node(node),
    };

    let mut nodes = deepex
        .nodes()
        .iter()
        .map(|node| -> ExResult<_> {
            let deepex_val = make_deepex(node.clone());
            let deepex_der = partial_deepex(var_idx, deepex_val.clone(), ops)?;
            Ok(Some(ValueDerivative {
                val: deepex_val,
                der: deepex_der,
            }))
        })
        .collect::<ExResult<Vec<_>>>()?;

    let partial_bin_ops_of_deepex = deepex
        .bin_ops()
        .reprs
        .iter()
        .map(|repr| -> ExResult<&PartialDerivative<'a, T>> {
            partial_derivative_ops
                .iter()
                .find(|pdo| &pdo.repr == repr)
                .ok_or(ExError {
                    msg: format!(
                        "derivative operator of {} needed for partial derivative",
                        repr
                    ),
                })
        })
        .collect::<ExResult<SmallVec<[&PartialDerivative<'a, T>; N_BINOPS_OF_DEEPEX_ON_STACK]>>>(
        )?;

    let mut num_inds = prio_indices.clone();
    let mut used_prio_indices = ExprIdxVec::new();

    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let node_1 = nodes[num_idx].take();
        let node_2 = nodes[num_idx + 1].take();

        let pd_deepex = if let (Some(n1), Some(n2)) = (node_1, node_2) {
            let pdo = &partial_bin_ops_of_deepex[bin_op_idx];
            pdo.bin_op.ok_or(ExError {
                msg: format!("cannot find binary op for {}", pdo.repr),
            })?(n1, n2, ops)
        } else {
            Err(ExError {
                msg: "nodes do not contain values in partial derivative".to_string(),
            })
        }?;
        nodes[num_idx] = Some(pd_deepex);
        nodes.remove(num_idx + 1);
        // reduce indices after removed position
        for num_idx_after in num_inds.iter_mut() {
            if *num_idx_after > num_idx {
                *num_idx_after -= 1;
            }
        }
        used_prio_indices.push(bin_op_idx);
    }
    let res = nodes[0]
        .take()
        .ok_or(ExError {
            msg: "node 0 needs to contain valder at the end of partial derviative".to_string(),
        })?
        .der;
    let (res, _) = res.var_names_union(deepex);
    Ok(res)
}

pub fn partial_deepex<'a, T: Float + Debug>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    ops: &[Operator<'a, T>],
) -> ExResult<DeepEx<'a, T>> {
    let partial_derivative_ops = make_partial_derivative_ops::<T>();
    let inner = partial_derivative_inner(var_idx, deepex.clone(), &partial_derivative_ops, ops)?;
    let outer = partial_derivative_outer(deepex, &partial_derivative_ops, ops)?;
    Ok(mul(inner, outer, find_as_bin_op_with_reprs("*", ops)?)?)
}

fn add<'a, T: Float + Debug>(
    summand_1: DeepEx<'a, T>,
    summand_2: DeepEx<'a, T>,
    add_op: BinOpsWithReprs<'a, T>,
) -> ExResult<DeepEx<'a, T>> {
    let (summand_1, summand_2) = summand_1.var_names_union(summand_2);
    Ok(if summand_1.is_zero() {
        summand_2
    } else if summand_2.is_zero() {
        summand_1
    } else {
        summand_1.operate_bin(summand_2, add_op)
    })
}

fn sub<'a, T: Float + Debug>(
    sub_1: DeepEx<'a, T>,
    sub_2: DeepEx<'a, T>,
    sub_op: BinOpsWithReprs<'a, T>,
) -> ExResult<DeepEx<'a, T>> {
    let (sub_1, sub_2) = sub_1.var_names_union(sub_2);
    Ok(if sub_2.is_zero() {
        sub_1
    } else {
        sub_1.operate_bin(sub_2, sub_op)
    })
}

fn mul<'a, T: Float + Debug>(
    factor_1: DeepEx<'a, T>,
    factor_2: DeepEx<'a, T>,
    mul_op: BinOpsWithReprs<'a, T>,
) -> ExResult<DeepEx<'a, T>> {
    let zero = DeepEx::zero();
    let (factor_1, factor_2) = factor_1.var_names_union(factor_2);
    let zero = zero.var_names_like_other(&factor_1);
    Ok(if factor_1.is_zero() || factor_2.is_zero() {
        zero
    } else if factor_1.is_one() {
        factor_2
    } else if factor_2.is_one() {
        factor_1
    } else {
        factor_1.operate_bin(factor_2, mul_op)
    })
}

fn div<'a, T: Float + Debug>(
    numerator: DeepEx<'a, T>,
    denominator: DeepEx<'a, T>,
    div_op: BinOpsWithReprs<'a, T>,
) -> ExResult<DeepEx<'a, T>> {
    let zero = DeepEx::zero();
    let (numerator, denominator) = numerator.var_names_union(denominator);
    let zero = zero.var_names_like_other(&numerator);
    if numerator.is_zero() && !denominator.is_zero() {
        Ok(zero)
    } else if denominator.is_one() {
        Ok(numerator)
    } else {
        Ok(numerator.operate_bin(denominator, div_op))
    }
}

fn pow<'a, T: Float + Debug>(
    base: DeepEx<'a, T>,
    exponent: DeepEx<'a, T>,
    power_op: BinOpsWithReprs<'a, T>,
) -> ExResult<DeepEx<'a, T>> {
    let zero = DeepEx::zero();
    let one = DeepEx::one();
    let (base, exponent) = base.var_names_union(exponent);
    let zero = zero.var_names_like_other(&base);
    let one = one.var_names_like_other(&base);
    Ok(if base.is_zero() && exponent.is_zero() {
        return Err(ExError {
            msg: "base and exponent both zero. help. fatal. ah. help.".to_string(),
        });
    } else if base.is_zero() {
        zero
    } else if exponent.is_zero() {
        one
    } else if exponent.is_one() {
        base
    } else {
        base.operate_bin(exponent, power_op)
    })
}

fn mul_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("*", &ops)
}
fn div_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("/", &ops)
}
fn add_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("+", &ops)
}
fn sub_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("-", &ops)
}
fn pow_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("^", &ops)
}
fn minus_find_unary<'a, T: Copy + Debug>(
    ops: &[Operator<'a, T>],
) -> ExResult<UnaryOpWithReprs<'a, T>> {
    find_as_unary_op_with_reprs("-", &ops)
}

pub fn make_partial_derivative_ops<'a, T: Float + Debug>() -> Vec<PartialDerivative<'a, T>> {
    vec![
        PartialDerivative {
            repr: "^",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> ExResult<ValueDerivative<T>> {
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let log_op = find_as_unary_op_with_reprs("log", ops)?;
                    let mul_op = mul_find(ops)?;
                    let add_op = add_find(ops)?;
                    let sub_op = sub_find(ops)?;

                    let one = DeepEx::one();
                    let val = pow(f.val.clone(), g.val.clone(), power_op.clone())?;
                    let g_minus_1 = g.val.clone().operate_bin(one, sub_op);
                    let der_1 = mul(
                        mul(
                            pow(f.val.clone(), g_minus_1, power_op.clone())?,
                            g.val.clone(),
                            mul_op.clone(),
                        )?,
                        f.der.clone(),
                        mul_op.clone(),
                    )?;

                    let der_2 = mul(
                        mul(val.clone(), f.val.operate_unary(log_op), mul_op.clone())?,
                        g.der.clone(),
                        mul_op,
                    )?;

                    let der = add(der_1, der_2, add_op)?;
                    Ok(ValueDerivative { val, der })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "+",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> ExResult<ValueDerivative<T>> {
                    let add_op = add_find(ops)?;

                    Ok(ValueDerivative {
                        val: add(f.val, g.val, add_op.clone())?,
                        der: add(f.der, g.der, add_op)?,
                    })
                },
            ),
            unary_outer_op: Some(
                |_: DeepEx<T>, _: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> { Ok(DeepEx::one()) },
            ),
        },
        PartialDerivative {
            repr: "-",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> ExResult<ValueDerivative<T>> {
                    let sub_op = sub_find(ops)?;

                    Ok(ValueDerivative {
                        val: sub(f.val, g.val, sub_op.clone())?,
                        der: sub(f.der, g.der, sub_op)?,
                    })
                },
            ),
            unary_outer_op: Some(
                |_: DeepEx<'a, T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<'a, T>> {
                    let one = DeepEx::one();
                    let minus = minus_find_unary(ops)?;
                    Ok(one.with_new_unary_op(minus))
                },
            ),
        },
        PartialDerivative {
            repr: "*",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> ExResult<ValueDerivative<T>> {
                    let mul_op = mul_find(ops)?;
                    let add_op = add_find(ops)?;

                    let val = mul(f.val.clone(), g.val.clone(), mul_op.clone())?;
                    let der_1 = mul(g.val, f.der, mul_op.clone())?;
                    let der_2 = mul(g.der, f.val, mul_op)?;
                    let der = add(der_1, der_2, add_op)?;
                    Ok(ValueDerivative { val, der })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "/",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> ExResult<ValueDerivative<T>> {
                    let mul_op = mul_find(ops)?;
                    let div_op = div_find(ops)?;
                    let sub_op = sub_find(ops)?;

                    let val = div(f.val.clone(), g.val.clone(), div_op.clone())?;

                    let numerator = sub(
                        mul(f.der, g.val.clone(), mul_op.clone())?,
                        mul(g.der, f.val, mul_op.clone())?,
                        sub_op,
                    )?;
                    let denominator = mul(g.val.clone(), g.val, mul_op)?;
                    Ok(ValueDerivative {
                        val,
                        der: div(numerator, denominator, div_op)?,
                    })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "sqrt",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<'a, T>> {
                    let mul_op = mul_find(ops)?;
                    let div_op = div_find(ops)?;
                    let one = DeepEx::one();
                    let two = DeepEx::num(T::from(2.0).unwrap());
                    div(one, mul(two, f, mul_op)?, div_op)
                },
            ),
        },
        PartialDerivative {
            repr: "log",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<'a, T>> {
                    let div_op = div_find(ops)?;
                    div(
                        DeepEx::one(),
                        f.with_new_unary_op(UnaryOpWithReprs::new()),
                        div_op,
                    )
                },
            ),
        },
        PartialDerivative {
            repr: "exp",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, _: &[Operator<'a, T>]| -> ExResult<DeepEx<'a, T>> { Ok(f) },
            ),
        },
        PartialDerivative {
            repr: "sin",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let unary_op = find_as_unary_op_with_reprs("cos", ops)?;
                    Ok(f.with_new_unary_op(unary_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cos",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let mut sin = find_as_unary_op_with_reprs("sin", ops)?;
                    let mut minus = minus_find_unary(ops)?;
                    sin.append_front(&mut minus);
                    Ok(f.with_new_unary_op(sin))
                },
            ),
        },
        PartialDerivative {
            repr: "tan",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<'a, T>> {
                    let cos_op = find_as_unary_op_with_reprs("cos", ops)?;
                    let power_op = pow_find(ops)?;
                    let div_op = div_find(ops)?;
                    let two = DeepEx::num(T::from(2.0).unwrap());
                    let cos_squared_ex = f
                        .clone()
                        .with_new_unary_op(cos_op)
                        .operate_bin(two, power_op);
                    div(DeepEx::one(), cos_squared_ex, div_op)
                },
            ),
        },
        PartialDerivative {
            repr: "asin",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let sqrt_op = find_as_unary_op_with_reprs("sqrt", ops)?;
                    let power_op = pow_find(ops)?;
                    let one = DeepEx::one();
                    let sub_op = sub_find(ops)?;
                    let div_op = div_find(ops)?;

                    let two = DeepEx::num(T::from(2.0).unwrap());
                    let inner_squared = f
                        .with_new_unary_op(UnaryOpWithReprs::new())
                        .operate_bin(two, power_op);
                    let insq_min1_sqrt =
                        sub(one.clone(), inner_squared, sub_op)?.operate_unary(sqrt_op);
                    div(one.clone(), insq_min1_sqrt, div_op)
                },
            ),
        },
        PartialDerivative {
            repr: "acos",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let sqrt_op = find_as_unary_op_with_reprs("sqrt", ops)?;
                    let power_op = pow_find(ops)?;
                    let minus_op = minus_find_unary(ops)?;
                    let sub_op = sub_find(ops)?;
                    let div_op = div_find(ops)?;

                    let one = DeepEx::one();
                    let two = DeepEx::num(T::from(2.0).unwrap());
                    let inner_squared = f
                        .with_new_unary_op(UnaryOpWithReprs::new())
                        .operate_bin(two, power_op);
                    let denominator =
                        sub(one.clone(), inner_squared, sub_op)?.operate_unary(sqrt_op);
                    Ok(div(one, denominator, div_op)?.operate_unary(minus_op))
                },
            ),
        },
        PartialDerivative {
            repr: "atan",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let power_op = pow_find(ops)?;
                    let add_op = add_find(ops)?;
                    let div_op = div_find(ops)?;
                    let one = DeepEx::one();
                    let two = DeepEx::num(T::from(2.0).unwrap());
                    let inner_squared =
                        pow(f.with_new_unary_op(UnaryOpWithReprs::new()), two, power_op)?;
                    div(one.clone(), add(one, inner_squared, add_op)?, div_op)
                },
            ),
        },
        PartialDerivative {
            repr: "sinh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let cosh_op = find_as_unary_op_with_reprs("cosh", ops)?;
                    Ok(f.with_new_unary_op(cosh_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cosh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let sinh_op = find_as_unary_op_with_reprs("sinh", ops)?;
                    Ok(f.with_new_unary_op(sinh_op))
                },
            ),
        },
        PartialDerivative {
            repr: "tanh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let one = DeepEx::one();
                    let pow_op = pow_find(ops)?;
                    let tanh_op = find_as_unary_op_with_reprs("tanh", ops)?;
                    let sub_op = sub_find(ops)?;
                    let two = DeepEx::num(T::from(2.0).unwrap());
                    sub(one, pow(f.with_new_unary_op(tanh_op), two, pow_op)?, sub_op)
                },
            ),
        },
    ]
}

#[cfg(test)]
use crate::{
    expression::flat::flatten, operators::make_default_operators, prelude::*,
    util::assert_float_eq_f64,
};

#[test]
fn test_partial() {
    let ops = make_default_operators::<f64>();
    let dut = DeepEx::<f64>::from_str("z*sin(x)+cos(y)^(sin(z))").unwrap();
    let d_z = partial_deepex(2, dut.clone(), &ops).unwrap();
    let flat = flatten(d_z);
    assert_float_eq_f64(
        flat.eval(&[-0.18961918881278095, -6.383306547710852, 3.1742139703464503])
            .unwrap(),
        -0.18346624475117082,
    );
    let dut = DeepEx::<f64>::from_str("sin(x)/x^2").unwrap();
    let d_x = partial_deepex(0, dut, &ops).unwrap();
    let flat = flatten(d_x);
    assert_float_eq_f64(
        flat.eval(&[-0.18961918881278095]).unwrap(),
        -27.977974668662565,
    );

    let dut = DeepEx::<f64>::from_str("x^y").unwrap();
    let d_x = partial_deepex(0, dut, &ops).unwrap();
    let flat = flatten(d_x);
    assert_float_eq_f64(flat.eval(&[7.5, 3.5]).unwrap(), 539.164392544148);
}

#[test]
fn test_partial_3_vars() {
    fn eval(deepex: DeepEx<f64>, vars: &[f64]) -> f64 {
        flatten(deepex).eval(vars).unwrap()
    }
    fn assert(s: &str, vars: &[f64], ref_vals: &[f64]) {
        let ops = make_default_operators::<f64>();
        let dut = DeepEx::<f64>::from_str(s).unwrap();
        let d_x = partial_deepex(0, dut.clone(), &ops).unwrap();
        assert_float_eq_f64(eval(d_x, vars), ref_vals[0]);
        let d_y = partial_deepex(1, dut.clone(), &ops).unwrap();
        assert_float_eq_f64(eval(d_y, vars), ref_vals[1]);
        let d_z = partial_deepex(2, dut.clone(), &ops).unwrap();
        assert_float_eq_f64(eval(d_z, vars), ref_vals[2]);
    }
    assert("x+y+z", &[2345.3, 4523.5, 1.2], &[1.0, 1.0, 1.0]);
    assert(
        "x^2+y^2+z^2",
        &[2345.3, 4523.5, 1.2],
        &[2345.3 * 2.0, 4523.5 * 2.0, 2.4],
    );
}

#[test]
fn test_partial_x2x() {
    let ops = make_default_operators::<f64>();
    let deepex = DeepEx::<f64>::from_str("x * 2 * x").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = flatten(derivative.clone()).eval(&[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = flatten(derivative).eval(&[1.0]).unwrap();
    assert_float_eq_f64(result, 4.0);
}

#[test]
fn test_partial_cos_squared() {
    let ops = make_default_operators::<f64>();
    let deepex = DeepEx::<f64>::from_str("cos(y) ^ 2").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = flatten(derivative.clone()).eval(&[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = flatten(derivative).eval(&[1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
}

#[test]
fn test_num_ops() {
    let ops = make_default_operators::<f64>();
    let mul_op = find_as_bin_op_with_reprs("*", &ops).unwrap();
    fn eval<'a>(deepex: &DeepEx<'a, f64>, vars: &[f64], val: f64) {
        assert_float_eq_f64(flatten(deepex.clone()).eval(vars).unwrap(), val);
    }
    fn check_shape<'a>(deepex: &DeepEx<'a, f64>, n_nodes: usize) {
        assert_eq!(deepex.nodes().len(), n_nodes);
        assert_eq!(deepex.bin_ops.ops.len(), n_nodes - 1);
        assert_eq!(deepex.bin_ops.reprs.len(), n_nodes - 1);
    }

    let minus_one = DeepEx::from_str("-1").unwrap();
    let one = mul(minus_one.clone(), minus_one.clone(), mul_op).unwrap();
    check_shape(&one, 1);
    eval(&one, &[], 1.0);
}

#[test]
fn test_partial_combined() {
    let ops = make_default_operators::<f64>();
    let deepex = DeepEx::<f64>::from_str("sin(x) + cos(y) ^ 2").unwrap();
    let d_y = partial_deepex(1, deepex.clone(), &ops).unwrap();
    let result = flatten(d_y.clone()).eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = flatten(d_y).eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
    let d_x = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = flatten(d_x.clone()).eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.5002954462477305);
    let result = flatten(d_x).eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, 0.8438539587324921);
}

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
fn test_partial_inner() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64], var_idx: usize) {
        let partial_derivative_ops = make_partial_derivative_ops::<f64>();
        let ops = make_default_operators::<f64>();
        let deepex_1 = DeepEx::<f64>::from_str(text).unwrap();
        let deri =
            partial_derivative_inner(var_idx, deepex_1, &partial_derivative_ops, &ops).unwrap();
        let flatex = flatten(deri);
        for i in 0..vals.len() {
            assert_float_eq_f64(flatex.eval(&[vals[i]]).unwrap(), ref_vals[i]);
        }
    }
    test("sin(x)", &[1.0, 0.0, 2.0], &[1.0, 1.0, 1.0], 0);
    test("sin(x^2)", &[1.0, 0.0, 2.0], &[2.0, 0.0, 4.0], 0);
}

#[test]
fn test_partial_outer() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64]) {
        let partial_derivative_ops = make_partial_derivative_ops::<f64>();
        let ops = make_default_operators::<f64>();
        let deepex_1 = DeepEx::<f64>::from_str(text).unwrap();
        let deepex = deepex_1.nodes()[0].clone();

        match deepex {
            DeepNode::Expr(e) => {
                let deri =
                    partial_derivative_outer(e.clone(), &partial_derivative_ops, &ops).unwrap();
                let flatex = flatten(deri);
                for i in 0..vals.len() {
                    assert_float_eq_f64(flatex.eval(&[vals[i]]).unwrap(), ref_vals[i]);
                }
            }
            _ => (),
        }
    }
    test("x", &[1.0, 0.0, 2.0], &[1.0, 0.0, 2.0]);
    test(
        "sin(x)",
        &[1.0, 0.0, 2.0],
        &[0.5403023058681398, 1.0, -0.4161468365471424],
    );
}

#[test]
fn test_partial_derivative_simple() {
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
