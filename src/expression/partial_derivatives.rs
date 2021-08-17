use crate::{
    definitions::N_BINOPS_OF_DEEPEX_ON_STACK,
    expression::{
        deep::{BinOpsWithReprs, DeepEx, DeepNode, ExprIdxVec, UnaryOpWithReprs},
        deep_details::{self, OverloadedOps},
    },
    operators::{Operator, UnaryOp},
    ExParseError,
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
        ) -> Result<ValueDerivative<'a, T>, ExParseError>,
    >,
    unary_outer_op:
        Option<fn(DeepEx<'a, T>, &[Operator<'a, T>]) -> Result<DeepEx<'a, T>, ExParseError>>,
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
        msg: format!("operator {} needed for outer partial derivative", repr),
    }
}

fn partial_derivative_outer<'a, T: Float + Debug>(
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
    overloaded_ops: OverloadedOps<'a, T>,
    ops: &[Operator<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let factorexes =
        deepex
            .unary_op()
            .reprs
            .iter()
            .map(|repr| -> Result<DeepEx<'a, T>, ExParseError> {
                let op = partial_derivative_ops
                    .iter()
                    .find(|pdo| &pdo.repr == repr)
                    .ok_or_else(|| make_op_missing_err(repr))?;
                let unary_deri_op = op.unary_outer_op.ok_or_else(|| make_op_missing_err(repr))?;

                unary_deri_op(deepex.clone(), ops)
            });
    let resex = factorexes.fold(
        Ok(DeepEx::one(overloaded_ops)),
        |dp1, dp2| -> Result<DeepEx<T>, ExParseError> { mul_num(dp1?, dp2?) },
    );
    resex
}

fn partial_derivative_inner<'a, T: Float + Debug>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    partial_derivative_ops: &[PartialDerivative<'a, T>],
    overloaded_ops: OverloadedOps<'a, T>,
    ops: &[Operator<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    // special case, partial derivative of only 1 node
    if deepex.nodes().len() == 1 {
        let res = match deepex.nodes()[0].clone() {
            DeepNode::Num(_) => DeepEx::zero(overloaded_ops.clone()),
            DeepNode::Var((var_i, _)) => {
                if var_i == var_idx {
                    DeepEx::one(overloaded_ops.clone())
                } else {
                    DeepEx::zero(overloaded_ops.clone())
                }
            }
            DeepNode::Expr(mut e) => {
                e.set_overloaded_ops(Some(overloaded_ops.clone()));
                partial_deepex(var_idx, e, ops)?
            }
        };
        let (res, _) = res.var_names_union(deepex);
        return Ok(res);
    }

    let prio_indices = deep_details::prioritized_indices(&deepex.bin_ops().ops, deepex.nodes());

    let make_deepex = |node: DeepNode<'a, T>| match node {
        DeepNode::Expr(mut e) => {
            e.set_overloaded_ops(Some(overloaded_ops.clone()));
            e
        }
        _ => DeepEx::from_node(node, overloaded_ops.clone()),
    };

    let mut nodes = deepex
        .nodes()
        .iter()
        .map(|node| -> Result<_, ExParseError> {
            let deepex_val = make_deepex(node.clone());
            let deepex_der = partial_deepex(var_idx, deepex_val.clone(), ops)?;
            Ok(Some(ValueDerivative {
                val: deepex_val,
                der: deepex_der,
            }))
        })
        .collect::<Result<Vec<_>, ExParseError>>()?;

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

    let mut num_inds = prio_indices.clone();
    let mut used_prio_indices = ExprIdxVec::new();

    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let node_1 = nodes[num_idx].take();
        let node_2 = nodes[num_idx + 1].take();

        let pd_deepex = if let (Some(n1), Some(n2)) = (node_1, node_2) {
            let pdo = &partial_bin_ops_of_deepex[bin_op_idx];
            pdo.bin_op.ok_or(ExParseError {
                msg: format!("cannot find binary op for {}", pdo.repr),
            })?(n1, n2, ops)
        } else {
            Err(ExParseError {
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
    let mut res = nodes[0]
        .take()
        .ok_or(ExParseError {
            msg: "node 0 needs to contain valder at the end of partial derviative".to_string(),
        })?
        .der;
    res.set_overloaded_ops(Some(overloaded_ops));
    let (res, _) = res.var_names_union(deepex);
    Ok(res)
}

pub fn partial_deepex<'a, T: Float + Debug>(
    var_idx: usize,
    deepex: DeepEx<'a, T>,
    ops: &[Operator<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let partial_derivative_ops = make_partial_derivative_ops::<T>();
    let overloaded_ops = deep_details::find_overloaded_ops(ops).ok_or(ExParseError {
        msg: "one of overloaded ops not found".to_string(),
    })?;

    let inner = partial_derivative_inner(
        var_idx,
        deepex.clone(),
        &partial_derivative_ops,
        overloaded_ops.clone(),
        ops,
    )?;
    let outer =
        partial_derivative_outer(deepex, &partial_derivative_ops, overloaded_ops.clone(), ops)?;
    let mut res = mul_num(inner, outer)?;
    res.compile();
    res.set_overloaded_ops(Some(overloaded_ops));
    Ok(res)
}

fn add_num<'a, T: Float + Debug>(
    summand_1: DeepEx<'a, T>,
    summand_2: DeepEx<'a, T>,
) -> Result<DeepEx<'a, T>, ExParseError> {
    let (summand_1, summand_2) = summand_1.var_names_union(summand_2);
    Ok(if summand_1.is_zero() {
        summand_2
    } else if summand_2.is_zero() {
        summand_1
    } else {
        summand_1 + summand_2
    })
}

fn sub_num<'a, T: Float + Debug>(
    sub_1: DeepEx<'a, T>,
    sub_2: DeepEx<'a, T>,
) -> Result<DeepEx<'a, T>, ExParseError> {
    let (sub_1, sub_2) = sub_1.var_names_union(sub_2);
    Ok(if sub_2.is_zero() {
        sub_1
    } else {
        sub_1 - sub_2
    })
}

fn mul_num<'a, T: Float + Debug>(
    factor_1: DeepEx<'a, T>,
    factor_2: DeepEx<'a, T>,
) -> Result<DeepEx<'a, T>, ExParseError> {
    let zero = DeepEx::zero(factor_1.unpack_and_clone_overloaded_ops()?);
    let (factor_1, factor_2) = factor_1.var_names_union(factor_2);
    let zero = zero.var_names_like_other(&factor_1);
    Ok(if factor_1.is_zero() || factor_2.is_zero() {
        zero
    } else if factor_1.is_one() {
        factor_2
    } else if factor_2.is_one() {
        factor_1
    } else {
        factor_1 * factor_2
    })
}

fn div_num<'a, T: Float + Debug>(
    numerator: DeepEx<'a, T>,
    denominator: DeepEx<'a, T>,
) -> Result<DeepEx<'a, T>, ExParseError> {
    let zero = DeepEx::zero(numerator.unpack_and_clone_overloaded_ops()?);
    let (numerator, denominator) = numerator.var_names_union(denominator);
    let zero = zero.var_names_like_other(&numerator);
    if numerator.is_zero() && !denominator.is_zero() {
        Ok(zero)
    } else if denominator.is_one() {
        Ok(numerator)
    } else {
        Ok(numerator / denominator)
    }
}

fn pow_num<'a, T: Float + Debug>(
    base: DeepEx<'a, T>,
    exponent: DeepEx<'a, T>,
    power_op: BinOpsWithReprs<'a, T>,
) -> Result<DeepEx<'a, T>, ExParseError> {
    let zero = DeepEx::zero(base.unpack_and_clone_overloaded_ops()?);
    let one = DeepEx::one(base.unpack_and_clone_overloaded_ops()?);
    let (base, exponent) = base.var_names_union(exponent);
    let zero = zero.var_names_like_other(&base);
    let one = one.var_names_like_other(&base);
    Ok(if base.is_zero() && exponent.is_zero() {
        return Err(ExParseError {
            msg: "base and exponent both zero. help. fatal. ah. help.".to_string(),
        });
    } else if base.is_zero() {
        zero
    } else if exponent.is_zero() {
        one
    } else {
        base.operate_bin(exponent, power_op)
    })
}

pub fn make_partial_derivative_ops<'a, T: Float + Debug>() -> Vec<PartialDerivative<'a, T>> {
    vec![
        PartialDerivative {
            repr: "^",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 ops: &[Operator<'a, T>]|
                 -> Result<ValueDerivative<T>, ExParseError> {
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let log_op = find_as_unary_op_with_reprs("log", ops)?;

                    let one = DeepEx::one_like(&f.val)?;
                    let val = pow_num(f.val.clone(), g.val.clone(), power_op.clone())?;

                    let der_1 = mul_num(
                        mul_num(
                            pow_num(f.val.clone(), g.val.clone() - one, power_op.clone())?,
                            g.val.clone(),
                        )?,
                        f.der.clone(),
                    )?;

                    let der_2 = mul_num(
                        mul_num(val.clone(), f.val.operate_unary(log_op))?,
                        g.der.clone(),
                    )?;

                    let der = add_num(der_1, der_2)?;
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
                 _: &[Operator<'a, T>]|
                 -> Result<ValueDerivative<T>, ExParseError> {
                    Ok(ValueDerivative {
                        val: add_num(f.val, g.val)?,
                        der: add_num(f.der, g.der)?,
                    })
                },
            ),
            unary_outer_op: Some(
                |f: DeepEx<T>, _: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    Ok(f.clone())
                },
            ),
        },
        PartialDerivative {
            repr: "-",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 _: &[Operator<'a, T>]|
                 -> Result<ValueDerivative<T>, ExParseError> {
                    Ok(ValueDerivative {
                        val: sub_num(f.val, g.val)?,
                        der: sub_num(f.der, g.der)?,
                    })
                },
            ),
            unary_outer_op: Some(
                |f: DeepEx<'a, T>,
                 ops: &[Operator<'a, T>]|
                 -> Result<DeepEx<'a, T>, ExParseError> {
                    let minus = find_as_unary_op_with_reprs("-", ops)?;
                    Ok(f.with_new_unary_op(minus))
                },
            ),
        },
        PartialDerivative {
            repr: "*",
            bin_op: Some(
                |f: ValueDerivative<T>,
                 g: ValueDerivative<T>,
                 _: &[Operator<'a, T>]|
                 -> Result<ValueDerivative<T>, ExParseError> {
                    let val = mul_num(f.val.clone(), g.val.clone())?;

                    let der_1 = mul_num(g.val, f.der)?;
                    let der_2 = mul_num(g.der, f.val)?;
                    let der = add_num(der_1, der_2)?;
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
                 _: &[Operator<'a, T>]|
                 -> Result<ValueDerivative<T>, ExParseError> {
                    let val = div_num(f.val.clone(), g.val.clone())?;

                    let numerator =
                        sub_num(mul_num(f.der, g.val.clone())?, mul_num(g.der, f.val)?)?;
                    let denominator = mul_num(g.val.clone(), g.val)?;
                    Ok(ValueDerivative {
                        val,
                        der: div_num(numerator, denominator)?,
                    })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "sqrt",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, _: &[Operator<'a, T>]| -> Result<DeepEx<'a, T>, ExParseError> {
                    let one = DeepEx::one_like(&f)?;
                    let two = one.clone() + one.clone();
                    Ok(one / (two * f))
                },
            ),
        },
        PartialDerivative {
            repr: "log",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, _: &[Operator<'a, T>]| -> Result<DeepEx<'a, T>, ExParseError> {
                    Ok(DeepEx::one_like(&f)? / f.with_new_unary_op(UnaryOpWithReprs::new()))
                },
            ),
        },
        PartialDerivative {
            repr: "exp",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>, _: &[Operator<'a, T>]| -> Result<DeepEx<'a, T>, ExParseError> {
                    Ok(f)
                },
            ),
        },
        PartialDerivative {
            repr: "sin",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let unary_op = find_as_unary_op_with_reprs("cos", ops)?;
                    Ok(f.with_new_unary_op(unary_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cos",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let mut sin = find_as_unary_op_with_reprs("sin", ops)?;
                    let mut minus = find_as_unary_op_with_reprs("-", ops)?;
                    sin.append_front(&mut minus);
                    Ok(f.with_new_unary_op(sin))
                },
            ),
        },
        PartialDerivative {
            repr: "tan",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T>,
                 ops: &[Operator<'a, T>]|
                 -> Result<DeepEx<'a, T>, ExParseError> {
                    let cos_op = find_as_unary_op_with_reprs("cos", ops)?;
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let two = DeepEx::one_like(&f)? + DeepEx::one_like(&f)?;
                    let cos_squared_ex = f
                        .clone()
                        .with_new_unary_op(cos_op)
                        .operate_bin(two, power_op);
                    Ok(DeepEx::one_like(&f)? / cos_squared_ex)
                },
            ),
        },
        PartialDerivative {
            repr: "asin",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let sqrt_op = find_as_unary_op_with_reprs("sqrt", ops)?;
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let one = DeepEx::one_like(&f)?;
                    let two = one.clone() + one.clone();
                    let inner_squared = f
                        .with_new_unary_op(UnaryOpWithReprs::new())
                        .operate_bin(two, power_op);
                    Ok(one.clone() / (one - inner_squared).operate_unary(sqrt_op))
                },
            ),
        },
        PartialDerivative {
            repr: "acos",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let sqrt_op = find_as_unary_op_with_reprs("sqrt", ops)?;
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let minus_op = find_as_unary_op_with_reprs("-", ops)?;

                    let one = DeepEx::one_like(&f)?;
                    let two = one.clone() + one.clone();
                    let inner_squared = f
                        .with_new_unary_op(UnaryOpWithReprs::new())
                        .operate_bin(two, power_op);
                    Ok((one.clone() / (one - inner_squared).operate_unary(sqrt_op))
                        .operate_unary(minus_op))
                },
            ),
        },
        PartialDerivative {
            repr: "atan",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let one = DeepEx::one_like(&f)?;
                    let two = one.clone() + one.clone();
                    let inner_squared = f
                        .with_new_unary_op(UnaryOpWithReprs::new())
                        .operate_bin(two, power_op);
                    Ok(one.clone() / (one + inner_squared))
                },
            ),
        },
        PartialDerivative {
            repr: "sinh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let cosh_op = find_as_unary_op_with_reprs("cosh", ops)?;
                    Ok(f.with_new_unary_op(cosh_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cosh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let sinh_op = find_as_unary_op_with_reprs("sinh", ops)?;
                    Ok(f.with_new_unary_op(sinh_op))
                },
            ),
        },
        PartialDerivative {
            repr: "tanh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> Result<DeepEx<T>, ExParseError> {
                    let one = DeepEx::one_like(&f)?;
                    let power_op = find_as_bin_op_with_reprs("^", ops)?;
                    let tanh_op = find_as_unary_op_with_reprs("tanh", ops)?;
                    let two = one.clone() + one.clone();
                    Ok(one - f.with_new_unary_op(tanh_op).operate_bin(two, power_op))
                },
            ),
        },
    ]
}

#[cfg(test)]
use {
    super::flat::flatten,
    crate::{operators::make_default_operators, util::assert_float_eq_f64},
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
        let ovops = deep_details::find_overloaded_ops(&ops).unwrap();
        match deepex_1.nodes()[0].clone() {
            DeepNode::Expr(e) => {
                let deri = partial_derivative_inner(
                    var_idx,
                    e.clone(),
                    &partial_derivative_ops,
                    ovops,
                    &ops,
                )
                .unwrap();

                let flatex = flatten(deri);
                for i in 0..vals.len() {
                    assert_float_eq_f64(flatex.eval(&[vals[i]]).unwrap(), ref_vals[i]);
                }
            }
            _ => panic!("test should not end up here"),
        };
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
        let ovops = deep_details::find_overloaded_ops(&ops).unwrap();

        match deepex {
            DeepNode::Expr(e) => {
                let deri =
                    partial_derivative_outer(e.clone(), &partial_derivative_ops, ovops, &ops)
                        .unwrap();
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
