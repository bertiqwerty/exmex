use std::{
    fmt::{Debug, Display},
    iter,
    str::FromStr,
};

use num::Float;
use smallvec::SmallVec;

use crate::{
    data_type::{DataType, NeutralElts},
    definitions::N_BINOPS_OF_DEEPEX_ON_STACK,
    expression::{
        deep::{find_unary_op, prioritized_indices, DeepEx, DeepNode, UnaryOpWithReprs},
        flat::ExprIdxVec,
    },
    format_exerr, ExError, ExResult, Express, MakeOperators, MatchLiteral, Operator,
};

pub fn check_partial_index(var_idx: usize, n_vars: usize, unparsed: &str) -> ExResult<()> {
    if var_idx >= n_vars {
        Err(format_exerr!(
            "index {} is invalid since we have only {} vars in {}",
            var_idx,
            n_vars,
            unparsed
        ))
    } else {
        Ok(())
    }
}
/// *`feature = "partial"`* - Trait for partial differentiation.  
pub trait Differentiate<'a, T: Clone>
where
    Self: Sized + Express<'a, T> + Display + Debug,
{
    /// *`feature = "partial"`* - This method computes a new expression
    /// that is the partial derivative of `self` with default operators.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    ///
    /// let expr = FlatEx::<f64>::parse("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.partial(0)?;
    ///
    /// assert!((dexpr_dx.eval(&[9e5, 2.0])? - (5.0 as f64).sin()).abs() < 1e-12);
    /// //                        |    
    /// //           The partial derivative dexpr_dx does depend on x. Still, it
    /// //           expects the same number of parameters as the corresponding
    /// //           antiderivative. Hence, you can pass any number for x.  
    ///
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    /// # Arguments
    ///
    /// * `var_idx` - variable with respect to which the partial derivative is computed
    ///
    /// # Errors
    ///
    /// * If you use custom operators this might not work as expected. It could return an [`ExError`](crate::ExError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial(self, var_idx: usize) -> ExResult<Self>
    where
        T: DataType + NeutralElts + Float,
        <T as FromStr>::Err: Debug,
    {
        self.partial_nth(var_idx, 1)
    }

    /// *`feature = "partial"`* - Computes the nth partial derivative with respect to one variable
    /// # Example
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    ///
    /// let mut expr = FlatEx::<f64>::parse("x^4+y^4")?;
    ///
    /// let dexpr_dxx_nth = expr.clone().partial_nth(0, 2)?;
    ///
    /// let dexpr_dx = expr.partial(0)?;
    /// let dexpr_dxx_2step = dexpr_dx.partial(0)?;
    ///
    /// assert!((dexpr_dxx_2step.eval(&[4.3, 2.1])? - dexpr_dxx_nth.eval(&[4.3, 2.1])?).abs() < 1e-12);
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    /// # Arguments
    ///
    /// * `var_idx` - variable with respect to which the partial derivative is computed
    /// * `n` - order of derivation
    ///
    /// # Errors
    ///
    /// * If you use custom operators this might not work as expected. It could return an [`ExError`](crate::ExError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial_nth(self, var_idx: usize, n: usize) -> ExResult<Self>
    where
        T: DataType + NeutralElts + Float,
        <T as FromStr>::Err: Debug,
    {
        self.partial_iter(iter::repeat(var_idx).take(n))
    }

    /// *`feature = "partial"`* - Computes a chain of partial derivatives with respect to the variables passed as iterator
    ///
    /// # Example
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::prelude::*;
    ///
    /// let mut expr = FlatEx::<f64>::parse("x^4+y^4")?;
    ///
    /// let dexpr_dxy_iter = expr.clone().partial_iter([0, 1].iter().copied())?;
    ///
    /// let dexpr_dx = expr.partial(0)?;
    /// let dexpr_dxy_2step = dexpr_dx.partial(1)?;
    ///
    /// assert!((dexpr_dxy_2step.eval(&[4.3, 2.1])? - dexpr_dxy_iter.eval(&[4.3, 2.1])?).abs() < 1e-12);
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    /// # Arguments
    ///
    /// * `var_idxs` - variables with respect to which the partial derivative is computed
    /// * `n` - order of derivation
    ///
    /// # Errors
    ///
    /// * If you use custom operators this might not work as expected. It could return an [`ExError`](crate::ExError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    fn partial_iter<I>(self, var_idxs: I) -> ExResult<Self>
    where
        T: DataType + NeutralElts + Float,
        <T as FromStr>::Err: Debug,
        I: Iterator<Item = usize> + Clone,
    {
        let mut deepex = self.to_deepex()?;

        let unparsed = deepex.unparse();
        for var_idx in var_idxs.clone() {
            check_partial_index(var_idx, deepex.var_names().len(), unparsed)?;
        }
        for var_idx in var_idxs {
            deepex = partial_deepex(var_idx, deepex)?;
        }
        deepex.compile();
        Self::from_deepex(deepex)
    }
}
#[derive(Clone)]
struct ValueDerivative<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    val: DeepEx<'a, T, OF, LM>,
    der: DeepEx<'a, T, OF, LM>,
}

type BinOpPartial<'a, T, OF, LM> = fn(
    ValueDerivative<'a, T, OF, LM>,
    ValueDerivative<'a, T, OF, LM>,
) -> ExResult<ValueDerivative<'a, T, OF, LM>>;

type UnaryOpOuter<'a, T, OF, LM> = fn(DeepEx<'a, T, OF, LM>) -> ExResult<DeepEx<'a, T, OF, LM>>;

pub struct PartialDerivative<'a, T: DataType, OF, LM>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    repr: &'a str,
    bin_op: Option<BinOpPartial<'a, T, OF, LM>>,
    unary_outer_op: Option<UnaryOpOuter<'a, T, OF, LM>>,
}

fn make_op_missing_err(repr: &str) -> ExError {
    format_exerr!("operator {} needed for outer partial derivative", repr)
}

fn partial_derivative_outer<'a, T: NeutralElts + DataType, OF, LM>(
    deepex: DeepEx<'a, T, OF, LM>,
    partial_derivative_ops: &[PartialDerivative<'a, T, OF, LM>],
) -> ExResult<DeepEx<'a, T, OF, LM>>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    let factorexes = deepex
        .unary_op()
        .reprs
        .iter()
        .enumerate()
        .map(|(idx, repr)| {
            let op = partial_derivative_ops
                .iter()
                .find(|pdo| pdo.repr == *repr)
                .ok_or_else(|| make_op_missing_err(repr))?;
            let unary_deri_op = op.unary_outer_op.ok_or_else(|| make_op_missing_err(repr))?;
            let mut new_deepex = deepex.clone();
            for _ in 0..idx {
                new_deepex = new_deepex.without_latest_unary_op();
            }
            unary_deri_op(new_deepex)
        });
    factorexes.fold(
        Ok(DeepEx::one()),
        |dp1, dp2| -> ExResult<DeepEx<T, OF, LM>> { dp1? * dp2? },
    )
}

fn partial_derivative_inner<'a, T: NeutralElts + Float + DataType, OF, LM>(
    var_idx: usize,
    deepex: DeepEx<'a, T, OF, LM>,
    partial_derivative_ops: &[PartialDerivative<'a, T, OF, LM>],
) -> ExResult<DeepEx<'a, T, OF, LM>>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
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
            DeepNode::Expr(e) => partial_deepex(var_idx, *e)?,
        };
        let (res, _) = res.var_names_union(deepex);
        return Ok(res);
    }

    let prio_indices = prioritized_indices(&deepex.bin_ops().ops, deepex.nodes());

    let make_deepex = |node: DeepNode<'a, T, OF, LM>| match node {
        DeepNode::Expr(e) => e,
        _ => Box::new(DeepEx::from_node(node)),
    };

    let mut nodes = deepex
        .nodes()
        .iter()
        .map(|node| -> ExResult<_> {
            let deepex_val = make_deepex(node.clone());
            let deepex_der = partial_deepex(var_idx, (*deepex_val).clone())?;
            Ok(Some(ValueDerivative {
                val: *deepex_val,
                der: deepex_der,
            }))
        })
        .collect::<ExResult<Vec<_>>>()?;

    let partial_bin_ops_of_deepex = deepex
        .bin_ops()
        .reprs
        .iter()
        .map(|repr| -> ExResult<&PartialDerivative<'a, T, OF, LM>> {
            partial_derivative_ops
                .iter()
                .find(|pdo| &pdo.repr == repr)
                .ok_or_else(|| {
                    format_exerr!(
                        "derivative operator of {} needed for partial derivative",
                        repr
                    )
                })
        })
        .collect::<ExResult<SmallVec<[&PartialDerivative<'a, T, OF, LM>; N_BINOPS_OF_DEEPEX_ON_STACK]>>>(
        )?;

    let mut num_inds = prio_indices.clone();
    let mut used_prio_indices = ExprIdxVec::new();

    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let node_1 = nodes[num_idx].take();
        let node_2 = nodes[num_idx + 1].take();

        let pd_deepex = if let (Some(n1), Some(n2)) = (node_1, node_2) {
            let pdo = &partial_bin_ops_of_deepex[bin_op_idx];
            pdo.bin_op
                .ok_or_else(|| format_exerr!("cannot find binary op for {}", pdo.repr))?(
                n1, n2
            )
        } else {
            Err(ExError::new(
                "nodes do not contain values in partial derivative",
            ))
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
        .ok_or_else(|| {
            ExError::new("node 0 needs to contain valder at the end of partial derviative")
        })?
        .der;
    let (res, _) = res.var_names_union(deepex);
    Ok(res)
}

pub fn partial_deepex<'a, T: NeutralElts + Float + DataType, OF, LM>(
    var_idx: usize,
    deepex: DeepEx<'a, T, OF, LM>,
) -> ExResult<DeepEx<'a, T, OF, LM>>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    let partial_derivative_ops = make_partial_derivative_ops::<T, OF, LM>();
    let inner = partial_derivative_inner(var_idx, deepex.clone(), &partial_derivative_ops)?;
    let outer = partial_derivative_outer(deepex, &partial_derivative_ops)?;
    inner * outer
}

fn find_minus_unary<'a, T: Clone + Debug>(
    ops: &[Operator<'a, T>],
) -> ExResult<UnaryOpWithReprs<'a, T>> {
    find_unary_op("-", ops)
}

enum Base {
    Two,
    Ten,
    Euler,
}
fn log_deri<'a, T: NeutralElts + Float + DataType, OF, LM>(
    f: DeepEx<'a, T, OF, LM>,
    base: Base,
) -> ExResult<DeepEx<'a, T, OF, LM>>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    let ln_base = |base_float: f64| DeepEx::from_num(T::from(base_float).unwrap().ln());
    let x = f.with_new_latest_unary_op(UnaryOpWithReprs::new());
    let denominator = match base {
        Base::Ten => (x * ln_base(10.0))?,
        Base::Two => (x * ln_base(2.0))?,
        Base::Euler => x,
    };
    DeepEx::one() / denominator
}

pub fn make_partial_derivative_ops<'a, T: NeutralElts + Float + DataType, OF, LM>(
) -> Vec<PartialDerivative<'a, T, OF, LM>>
where
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    vec![
        PartialDerivative {
            repr: "^",
            bin_op: Some(
                |f: ValueDerivative<T, OF, LM>,
                 g: ValueDerivative<T, OF, LM>|
                 -> ExResult<ValueDerivative<T, OF, LM>> {
                    let one = DeepEx::one();
                    let val = f.val.clone().pow(g.val.clone())?;
                    let g_minus_1 = (g.val.clone() - one)?;
                    let der_1 = ((f.val.clone().pow(g_minus_1)? * g.val.clone())? * f.der.clone())?;

                    let der_2 = ((val.clone() * f.val.ln()?)? * g.der.clone())?;

                    let der = (der_1 + der_2)?;
                    Ok(ValueDerivative { val, der })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "+",
            bin_op: Some(
                |f: ValueDerivative<T, OF, LM>,
                 g: ValueDerivative<T, OF, LM>|
                 -> ExResult<ValueDerivative<T, OF, LM>> {
                    Ok(ValueDerivative {
                        val: (f.val + g.val)?,
                        der: (f.der + g.der)?,
                    })
                },
            ),
            unary_outer_op: Some(|_: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                Ok(DeepEx::one())
            }),
        },
        PartialDerivative {
            repr: "-",
            bin_op: Some(
                |f: ValueDerivative<T, OF, LM>,
                 g: ValueDerivative<T, OF, LM>|
                 -> ExResult<ValueDerivative<T, OF, LM>> {
                    Ok(ValueDerivative {
                        val: (f.val - g.val)?,
                        der: (f.der - g.der)?,
                    })
                },
            ),
            unary_outer_op: Some(
                |_: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    let one = DeepEx::one();
                    let minus = find_minus_unary(one.ops())?;
                    Ok(one.with_only_unary_op(minus))
                },
            ),
        },
        PartialDerivative {
            repr: "*",
            bin_op: Some(
                |f: ValueDerivative<T, OF, LM>,
                 g: ValueDerivative<T, OF, LM>|
                 -> ExResult<ValueDerivative<T, OF, LM>> {
                    let val = (f.val.clone() * g.val.clone())?;
                    let der_1 = (g.val * f.der)?;
                    let der_2 = (g.der * f.val)?;
                    let der = (der_1 + der_2)?;
                    Ok(ValueDerivative { val, der })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "/",
            bin_op: Some(
                |f: ValueDerivative<T, OF, LM>,
                 g: ValueDerivative<T, OF, LM>|
                 -> ExResult<ValueDerivative<T, OF, LM>> {
                    let val = (f.val.clone() / g.val.clone())?;

                    let numerator = ((f.der * g.val.clone())? - (g.der * f.val)?)?;
                    let denominator = (g.val.clone() * g.val)?;
                    Ok(ValueDerivative {
                        val,
                        der: (numerator / denominator)?,
                    })
                },
            ),
            unary_outer_op: None,
        },
        PartialDerivative {
            repr: "sqrt",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    let one = DeepEx::one();
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    one / (two * f)?
                },
            ),
        },
        PartialDerivative {
            repr: "ln",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    log_deri(f, Base::Euler)
                },
            ),
        },
        PartialDerivative {
            repr: "log10",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    log_deri(f, Base::Ten)
                },
            ),
        },
        PartialDerivative {
            repr: "log2",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    log_deri(f, Base::Two)
                },
            ),
        },
        PartialDerivative {
            repr: "exp",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> { Ok(f) },
            ),
        },
        PartialDerivative {
            repr: "sin",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let unary_op = find_unary_op("cos", f.ops())?;
                Ok(f.with_new_latest_unary_op(unary_op))
            }),
        },
        PartialDerivative {
            repr: "cos",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let mut sin = find_unary_op("sin", f.ops())?;
                let minus = find_minus_unary(f.ops())?;
                sin.append_after(minus);
                Ok(f.with_new_latest_unary_op(sin))
            }),
        },
        PartialDerivative {
            repr: "tan",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<'a, T, OF, LM>| -> ExResult<DeepEx<'a, T, OF, LM>> {
                    let cos_op = find_unary_op("cos", f.ops())?;
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    let cos_squared_ex = f.clone().with_new_latest_unary_op(cos_op).pow(two)?;
                    DeepEx::one() / cos_squared_ex
                },
            ),
        },
        PartialDerivative {
            repr: "asin",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let sqrt_op = find_unary_op("sqrt", f.ops())?;
                let one = DeepEx::one();

                let two = DeepEx::from_num(T::from(2.0).unwrap());
                let inner_squared = f
                    .with_new_latest_unary_op(UnaryOpWithReprs::new())
                    .pow(two)?;
                let insq_min1_sqrt =
                    (one.clone() - inner_squared)?.operate_unary_opwithrepr(sqrt_op);
                one.clone() / insq_min1_sqrt
            }),
        },
        PartialDerivative {
            repr: "acos",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let sqrt_op = find_unary_op("sqrt", f.ops())?;
                let minus_op = find_minus_unary(f.ops())?;

                let one = DeepEx::one();
                let two = DeepEx::from_num(T::from(2.0).unwrap());
                let inner_squared = f
                    .with_new_latest_unary_op(UnaryOpWithReprs::new())
                    .pow(two)?;
                let denominator = (one.clone() - inner_squared)?.operate_unary_opwithrepr(sqrt_op);
                Ok((one / denominator)?.operate_unary_opwithrepr(minus_op))
            }),
        },
        PartialDerivative {
            repr: "atan",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let one = DeepEx::one();
                let two = DeepEx::from_num(T::from(2.0).unwrap());
                let inner_squared = f
                    .with_new_latest_unary_op(UnaryOpWithReprs::new())
                    .pow(two)?;
                one.clone() / (one + inner_squared)?
            }),
        },
        PartialDerivative {
            repr: "sinh",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let cosh_op = find_unary_op("cosh", f.ops())?;
                Ok(f.with_new_latest_unary_op(cosh_op))
            }),
        },
        PartialDerivative {
            repr: "cosh",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let sinh_op = find_unary_op("sinh", f.ops())?;
                Ok(f.with_new_latest_unary_op(sinh_op))
            }),
        },
        PartialDerivative {
            repr: "tanh",
            bin_op: None,
            unary_outer_op: Some(|f: DeepEx<T, OF, LM>| -> ExResult<DeepEx<T, OF, LM>> {
                let one = DeepEx::one();
                let tanh_op = find_unary_op("tanh", f.ops())?;
                let two = DeepEx::from_num(T::from(2.0).unwrap());
                one - f.with_new_latest_unary_op(tanh_op).pow(two)?
            }),
        },
    ]
}

#[cfg(test)]
use crate::{util::assert_float_eq_f64, FlatEx, FloatOpsFactory, NumberMatcher};

#[test]
fn test_pmp() -> ExResult<()> {
    let x = 1.5f64;
    let fex = FlatEx::<f64>::parse("+-+x")?;
    let deri = fex.partial(0)?;
    println!("{}", deri);
    let reference = -1.0;
    assert_float_eq_f64(deri.eval(&[x])?, reference);
    Ok(())
}
#[test]
fn test_compile() -> ExResult<()> {
    let deepex = DeepEx::<f64>::parse("1+(((a+x^2*x^2)))")?;
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "1.0+({a}+{x}^2.0*{x}^2.0)");
    let mut ddeepex = partial_deepex(1, deepex)?;
    ddeepex.compile();
    println!("{}", ddeepex);
    assert_eq!(
        format!("{}", ddeepex),
        "(({x}^2.0)*({x}*2.0))+(({x}*2.0)*({x}^2.0))"
    );
    Ok(())
}
#[test]
fn test_sincosin() -> ExResult<()> {
    let x = 1.5f64;
    let fex = FlatEx::<f64>::parse("sin(cos(sin(x)))")?;
    let deri = fex.partial(0)?;
    println!("{}", deri);
    let reference = x.cos() * (-x.sin().sin()) * x.sin().cos().cos();
    assert_float_eq_f64(deri.eval(&[x])?, reference);
    Ok(())
}

#[test]
fn test_partial() {
    let dut = DeepEx::<f64>::parse("z*sin(x)+cos(y)^(sin(z))").unwrap();
    let d_z = partial_deepex(2, dut.clone()).unwrap();
    assert_float_eq_f64(
        d_z.eval(&[-0.18961918881278095, -6.383306547710852, 3.1742139703464503])
            .unwrap(),
        -0.18346624475117082,
    );
    let dut = DeepEx::<f64>::parse("sin(x)/x^2").unwrap();
    let d_x = partial_deepex(0, dut).unwrap();
    assert_float_eq_f64(
        d_x.eval(&[-0.18961918881278095]).unwrap(),
        -27.977974668662565,
    );

    let dut = DeepEx::<f64>::parse("x^y").unwrap();
    let d_x = partial_deepex(0, dut).unwrap();
    assert_float_eq_f64(d_x.eval(&[7.5, 3.5]).unwrap(), 539.164392544148);
}

#[test]
fn test_partial_3_vars() {
    fn eval_(deepex: &DeepEx<f64, FloatOpsFactory<f64>, NumberMatcher>, vars: &[f64]) -> f64 {
        deepex.eval(vars).unwrap()
    }
    fn assert(s: &str, vars: &[f64], ref_vals: &[f64]) {
        let dut = DeepEx::<f64>::parse(s).unwrap();
        let d_x = partial_deepex(0, dut.clone()).unwrap();
        assert_float_eq_f64(eval_(&d_x, vars), ref_vals[0]);
        let d_y = partial_deepex(1, dut.clone()).unwrap();
        assert_float_eq_f64(eval_(&d_y, vars), ref_vals[1]);
        let d_z = partial_deepex(2, dut.clone()).unwrap();
        assert_float_eq_f64(eval_(&d_z, vars), ref_vals[2]);
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
    let deepex = DeepEx::<f64>::parse("x * 2 * x").unwrap();
    let derivative = partial_deepex(0, deepex.clone()).unwrap();
    let result = derivative.eval(&[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = derivative.eval(&[1.0]).unwrap();
    assert_float_eq_f64(result, 4.0);
}

#[test]
fn test_partial_cos_squared() {
    let deepex = DeepEx::<f64>::parse("cos(y) ^ 2").unwrap();
    let derivative = partial_deepex(0, deepex.clone()).unwrap();
    let result = derivative.eval(&[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = derivative.eval(&[1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
}

#[test]
fn test_num_ops() {
    fn eval_<'a>(
        deepex: &DeepEx<'a, f64, FloatOpsFactory<f64>, NumberMatcher>,
        vars: &[f64],
        val: f64,
    ) {
        assert_float_eq_f64(deepex.eval(vars).unwrap(), val);
    }
    fn check_shape(deepex: &DeepEx<f64, FloatOpsFactory<f64>, NumberMatcher>, n_nodes: usize) {
        assert_eq!(deepex.nodes().len(), n_nodes);
        assert_eq!(deepex.bin_ops().ops.len(), n_nodes - 1);
        assert_eq!(deepex.bin_ops().reprs.len(), n_nodes - 1);
    }

    let minus_one = DeepEx::<f64>::parse("-1").unwrap();
    let one = (minus_one.clone() * minus_one.clone()).unwrap();
    check_shape(&one, 1);
    eval_(&one, &[], 1.0);
}

#[test]
fn test_partial_combined() {
    let deepex = DeepEx::<f64>::parse("sin(x) + cos(y) ^ 2").unwrap();
    let d_y = partial_deepex(1, deepex.clone()).unwrap();
    let result = d_y.eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = d_y.eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
    let d_x = partial_deepex(0, deepex.clone()).unwrap();
    let result = d_x.eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.5002954462477305);
    let result = d_x.eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, 0.8438539587324921);
}

#[test]
fn test_partial_derivative_second_var() {
    let deepex = DeepEx::<f64>::parse("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(1, deepex.clone()).unwrap();
    let result = derivative.eval(&[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = derivative.eval(&[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.8414709848078965);
}

#[test]
fn test_partial_derivative_first_var() {
    let deepex = DeepEx::<f64>::parse("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(0, deepex.clone()).unwrap();
    let result = derivative.eval(&[0.0, 2345.03]).unwrap();
    assert_float_eq_f64(result, 1.0);
    let result = derivative.eval(&[1.0, 43212.43]).unwrap();
    assert_float_eq_f64(result, 0.5403023058681398);
}

#[test]
fn test_partial_inner() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64], var_idx: usize) {
        let partial_derivative_ops =
            make_partial_derivative_ops::<f64, FloatOpsFactory<f64>, NumberMatcher>();
        let deepex_1 = DeepEx::<f64>::parse(text).unwrap();
        let deri = partial_derivative_inner(var_idx, deepex_1, &partial_derivative_ops).unwrap();
        for i in 0..vals.len() {
            assert_float_eq_f64(deri.eval(&[vals[i]]).unwrap(), ref_vals[i]);
        }
    }
    test("sin(x)", &[1.0, 0.0, 2.0], &[1.0, 1.0, 1.0], 0);
    test("sin(x^2)", &[1.0, 0.0, 2.0], &[2.0, 0.0, 4.0], 0);
}

#[test]
fn test_partial_outer() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64]) {
        let partial_derivative_ops =
            make_partial_derivative_ops::<f64, FloatOpsFactory<f64>, NumberMatcher>();
        let deepex_1 = DeepEx::<f64>::parse(text).unwrap();
        let deepex = deepex_1.nodes()[0].clone();

        if let DeepNode::Expr(e) = deepex {
            let deri = partial_derivative_outer(*e, &partial_derivative_ops).unwrap();
            for i in 0..vals.len() {
                assert_float_eq_f64(deri.eval(&[vals[i]]).unwrap(), ref_vals[i]);
            }
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
fn test_partial_derivative_simple() -> ExResult<()> {
    let deepex = DeepEx::<f64>::parse("1")?;
    let derivative = partial_deepex(0, deepex)?;

    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 0.0),
        _ => unreachable!(),
    }
    let deepex = DeepEx::<f64>::parse("x")?;
    let derivative = partial_deepex(0, deepex)?;
    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 1.0),
        _ => unreachable!(),
    }
    let deepex = DeepEx::<f64>::parse("x^2")?;
    let derivative = partial_deepex(0, deepex)?;
    let result = derivative.eval(&[4.5])?;
    assert_float_eq_f64(result, 9.0);

    let deepex = DeepEx::<f64>::parse("sin(x)")?;
    let derivative = partial_deepex(0, deepex.clone())?;
    let result = derivative.eval(&[0.0])?;
    assert_float_eq_f64(result, 1.0);
    let result = derivative.eval(&[1.0])?;
    assert_float_eq_f64(result, 0.5403023058681398);
    Ok(())
}
