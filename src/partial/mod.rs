use std::{
    fmt::{self, Debug, Display, Formatter},
    str::FromStr,
};

use num::Float;
use smallvec::SmallVec;

use crate::{
    data_type::DataType,
    definitions::{N_BINOPS_OF_DEEPEX_ON_STACK, N_NODES_ON_STACK, N_VARS_ON_STACK},
    expression::flat::ExprIdxVec,
    format_exerr,
    operators::UnaryOp,
    BinOp, ExError, ExResult, Express, MakeOperators, Operator,
};
pub use details::{BinOpsWithReprs, UnaryOpWithReprs};

#[cfg(test)]
use crate::parser;

mod details;
/// *`feature = "partial"`* - Trait for partial differentiation.  
pub trait Differentiate<T: Clone>
where
    Self: Sized + Express<T> + Display + Debug,
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
    /// let mut expr = FlatEx::<f64>::from_str("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.partial(0)?;
    /// let dexpr_dy = expr.partial(1)?;
    ///
    /// assert!((dexpr_dx.eval(&[9e5, 2.0])? - (5.0 as f64).sin()).abs() < 1e-12);
    /// //             |    
    /// //           The partial derivative dexpr_dx does depend on x. Still, it
    /// //           expects the same number of parameters as the corresponding
    /// //           antiderivative. Hence, you can pass any number for x.  
    ///
    /// assert!((dexpr_dy.eval(&[2.5, 2.0])? - 10.0 * (5.0 as f64).cos()).abs() < 1e-12);
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
    fn partial(&self, var_idx: usize) -> ExResult<Self>
    where
        T: DataType + Float,
        <T as FromStr>::Err: Debug,
    {
        let ops = Self::OperatorFactory::make();
        let deepex = self.to_deepex(&ops)?;

        let unparsed = deepex.unparse();
        details::check_partial_index(var_idx, self.var_names().len(), unparsed.as_str())?;
        let d_i = partial_deepex(var_idx, deepex, &ops)?;
        Self::from_deepex(d_i, &ops)
    }

    /// *`feature = "partial"`* - Every trait implementation needs to implement the conversion to a deep
    /// expression to be able to use the default implementation of [`partial`](Differentiate::partial).
    fn to_deepex<'a>(&'a self, ops: &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug;

    /// *`feature = "partial"`* - Every trait implementation needs to implement the conversion from
    /// a deep expression to be able to use the default implementation of [`partial`](Differentiate::partial).
    fn from_deepex(deepex: DeepEx<T>, ops: &[Operator<T>]) -> ExResult<Self>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug;
}

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;

/// Correction for cases where nodes are unnecessarily wrapped in expression-nodes.
fn lift_nodes<T: Clone + Debug>(deepex: &mut DeepEx<T>) {
    if deepex.nodes().len() == 1 && deepex.unary_op().op.len() == 0 {
        if let DeepNode::Expr(e) = &deepex.nodes()[0] {
            *deepex = (**e).clone();
        }
    } else {
        for node in &mut deepex.nodes {
            if let DeepNode::Expr(e) = node {
                if e.nodes.len() == 1 && e.unary_op.op.len() == 0 {
                    match &mut e.nodes[0] {
                        DeepNode::Num(n) => *node = DeepNode::Num(n.clone()),
                        DeepNode::Var(v) => {
                            *node = DeepNode::Var(*v);
                        }
                        DeepNode::Expr(e_deeper) => {
                            lift_nodes(e_deeper);
                            if e_deeper.nodes.len() == 1 && e_deeper.unary_op.op.len() == 0 {
                                *node = DeepNode::Expr(e_deeper.clone());
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DeepNode<'a, T: Clone + Debug> {
    /// Boxing this due to <https://rust-lang.github.io/rust-clippy/master/index.html#large_enum_variant>
    Expr(Box<DeepEx<'a, T>>),
    Num(T),
    /// The contained integer points to the index of the variable.
    Var((usize, &'a str)),
}
impl<'a, T: Debug> DeepNode<'a, T>
where
    T: Clone + Debug + Float,
{
    fn zero() -> Self {
        DeepNode::Num(T::from(0.0).unwrap())
    }
    fn one() -> Self {
        DeepNode::Num(T::from(1.0).unwrap())
    }
    fn num(n: T) -> Self {
        DeepNode::Num(n)
    }
}
impl<'a, T: Clone + Debug> Debug for DeepNode<'a, T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DeepNode::Expr(e) => write!(f, "{}", e),
            DeepNode::Num(n) => write!(f, "{:?}", n),
            DeepNode::Var((_, var_name)) => write!(f, "{}", var_name),
        }
    }
}

/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T: Clone + Debug> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: BinOpsWithReprs<'a, T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: UnaryOpWithReprs<'a, T>,
    var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
}

impl<'a, T> DeepEx<'a, T>
where
    T: Clone + Debug,
{
    /// Compiles expression, needed for partial differentation.
    pub fn compile(&mut self) {
        lift_nodes(self);

        let prio_indices = details::prioritized_indices(&self.bin_ops.ops, &self.nodes);
        let mut num_inds = prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();

        let mut already_declined: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec::smallvec![false; self.nodes.len()];

        for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (DeepNode::Num(num_1), DeepNode::Num(num_2)) = (node_1, node_2) {
                if !(already_declined[num_idx] || already_declined[num_idx + 1]) {
                    let bin_op_result =
                        (self.bin_ops.ops[bin_op_idx].apply)(num_1.clone(), num_2.clone());
                    self.nodes[num_idx] = DeepNode::Num(bin_op_result);
                    self.nodes.remove(num_idx + 1);
                    already_declined.remove(num_idx + 1);
                    // reduce indices after removed position
                    for num_idx_after in num_inds.iter_mut() {
                        if *num_idx_after > num_idx {
                            *num_idx_after -= 1;
                        }
                    }
                    used_prio_indices.push(bin_op_idx);
                }
            } else {
                already_declined[num_idx] = true;
                already_declined[num_idx + 1] = true;
            }
        }

        let mut resulting_reprs = smallvec::smallvec![];
        self.bin_ops.ops = self
            .bin_ops
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|(i, bin_op)| {
                resulting_reprs.push(self.bin_ops.reprs[i]);
                bin_op.clone()
            })
            .collect();
        self.bin_ops.reprs = resulting_reprs;

        if self.nodes.len() == 1 {
            if let DeepNode::Num(n) = self.nodes[0].clone() {
                self.nodes[0] = DeepNode::Num(self.unary_op.op.apply(n));
                self.unary_op.op.clear();
                self.unary_op.reprs.clear();
            }
        }
    }

    pub fn new(
        nodes: Vec<DeepNode<'a, T>>,
        bin_ops: BinOpsWithReprs<'a, T>,
        unary_op: UnaryOpWithReprs<'a, T>,
    ) -> ExResult<DeepEx<'a, T>> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(format_exerr!(
                "mismatch between number of nodes {:?} and binary operators {:?} ({} vs {})",
                nodes,
                bin_ops.ops,
                nodes.len(),
                bin_ops.ops.len()
            ))
        } else {
            let mut found_vars = SmallVec::<[&str; N_VARS_ON_STACK]>::new();
            for node in &nodes {
                match node {
                    DeepNode::Num(_) => (),
                    DeepNode::Var((_, name)) => {
                        if !found_vars.contains(name) {
                            found_vars.push(name);
                        }
                    }
                    DeepNode::Expr(e) => {
                        for name in &e.var_names {
                            if !found_vars.contains(name) {
                                found_vars.push(name);
                            }
                        }
                    }
                }
            }
            found_vars.sort_unstable();
            let mut expr = DeepEx {
                nodes,
                bin_ops,
                unary_op,
                var_names: found_vars,
            };
            expr.compile();
            Ok(expr)
        }
    }

    fn from_node(node: DeepNode<'a, T>) -> DeepEx<'a, T> {
        DeepEx::new(vec![node], BinOpsWithReprs::new(), UnaryOpWithReprs::new()).unwrap()
    }

    fn one() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::one())
    }

    fn zero() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::zero())
    }

    fn from_num(x: T) -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::num(x))
    }

    fn with_new_latest_unary_op(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.remove_latest();
        self.unary_op.append_after(unary_op);
        self
    }

    fn with_only_unary_op(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.clear();
        self.unary_op.append_after(unary_op);
        self
    }

    pub fn bin_ops(&self) -> &BinOpsWithReprs<'a, T> {
        &self.bin_ops
    }

    pub fn unary_op(&self) -> &UnaryOpWithReprs<'a, T> {
        &self.unary_op
    }

    pub fn nodes(&self) -> &Vec<DeepNode<'a, T>> {
        &self.nodes
    }

    fn is_num(&self, num: T) -> bool
    where
        T: Float,
    {
        details::is_num(self, num)
    }

    fn is_one(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(1.0).unwrap())
    }

    fn is_zero(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(0.0).unwrap())
    }

    pub fn reset_vars(&mut self, new_var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>) {
        for node in &mut self.nodes {
            match node {
                DeepNode::Expr(e) => e.reset_vars(new_var_names.clone()),
                DeepNode::Var((i, var_name)) => {
                    for (new_idx, new_name) in new_var_names.iter().enumerate() {
                        if var_name == new_name {
                            *i = new_idx;
                        }
                    }
                }
                _ => (),
            }
        }
        self.var_names = new_var_names;
    }

    pub fn var_names(&self) -> &[&str] {
        &self.var_names
    }

    pub fn var_names_union(self, other: Self) -> (Self, Self) {
        let mut all_var_names = self.var_names.iter().copied().collect::<SmallVec<_>>();
        for name in other.var_names.clone() {
            if !all_var_names.contains(&name) {
                all_var_names.push(name);
            }
        }
        all_var_names.sort_unstable();
        let mut self_vars_updated = self;
        let mut other_vars_updated = other;
        self_vars_updated.reset_vars(all_var_names.clone());
        other_vars_updated.reset_vars(all_var_names);
        (self_vars_updated, other_vars_updated)
    }

    fn var_names_like_other(mut self, other: &Self) -> Self {
        self.var_names = other.var_names.clone();
        self
    }

    /// Applies a binary operator to self and other
    fn operate_bin(self, other: Self, bin_op: BinOpsWithReprs<'a, T>) -> Self {
        details::operate_bin(self, other, bin_op)
    }

    /// Applies a unary operator to self
    fn operate_unary(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.append_after(unary_op);
        self.compile();
        self
    }

    pub fn unparse(&self) -> String {
        details::unparse_raw::<T>(self)
    }
}

impl<'a, T: Clone + Debug> Display for DeepEx<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.unparse())
    }
}

#[derive(Clone)]
struct ValueDerivative<'a, T: Copy + Debug> {
    val: DeepEx<'a, T>,
    der: DeepEx<'a, T>,
}

pub fn find_op<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Option<Operator<'a, T>> {
    ops.iter().cloned().find(|op| op.repr() == repr)
}

type BinOpPartial<'a, T> = fn(
    ValueDerivative<'a, T>,
    ValueDerivative<'a, T>,
    &[Operator<'a, T>],
) -> ExResult<ValueDerivative<'a, T>>;

type UnaryOpOuter<'a, T> = fn(DeepEx<'a, T>, &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>;

pub struct PartialDerivative<'a, T: Copy + Debug> {
    repr: &'a str,
    bin_op: Option<BinOpPartial<'a, T>>,
    unary_outer_op: Option<UnaryOpOuter<'a, T>>,
}

fn find_as_bin_op_with_reprs<'a, T: Copy + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> ExResult<BinOpsWithReprs<'a, T>> {
    let op = find_op(repr, ops).ok_or(ExError {
        msg: format!("did not find operator {}", repr),
    })?;
    Ok(BinOpsWithReprs {
        reprs: smallvec::smallvec![op.repr()],
        ops: smallvec::smallvec![op.bin()?],
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
        reprs: smallvec::smallvec![op.repr()],
        op: UnaryOp::from_vec(smallvec::smallvec![op.unary()?]),
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
        .enumerate()
        .map(|(idx, repr)| {
            let op = partial_derivative_ops
                .iter()
                .find(|pdo| pdo.repr == *repr)
                .ok_or_else(|| make_op_missing_err(repr))?;
            let unary_deri_op = op.unary_outer_op.ok_or_else(|| make_op_missing_err(repr))?;
            let mut new_deepex = deepex.clone();
            for _ in 0..idx {
                new_deepex.unary_op.remove_latest();
            }
            unary_deri_op(new_deepex, ops)
        });
    let mul_op = mul_find(ops)?;
    factorexes.fold(Ok(DeepEx::one()), |dp1, dp2| -> ExResult<DeepEx<T>> {
        mul(dp1?, dp2?, mul_op.clone())
    })
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
            DeepNode::Expr(e) => partial_deepex(var_idx, *e, ops)?,
        };
        let (res, _) = res.var_names_union(deepex);
        return Ok(res);
    }

    let prio_indices = details::prioritized_indices(&deepex.bin_ops().ops, deepex.nodes());

    let make_deepex = |node: DeepNode<'a, T>| match node {
        DeepNode::Expr(e) => e,
        _ => Box::new(DeepEx::from_node(node)),
    };

    let mut nodes = deepex
        .nodes()
        .iter()
        .map(|node| -> ExResult<_> {
            let deepex_val = make_deepex(node.clone());
            let deepex_der = partial_deepex(var_idx, (*deepex_val).clone(), ops)?;
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
    mul(inner, outer, mul_find(ops)?)
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
    find_as_bin_op_with_reprs("*", ops)
}
fn div_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("/", ops)
}
fn add_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("+", ops)
}
fn sub_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("-", ops)
}
fn pow_find<'a, T: Copy + Debug>(ops: &[Operator<'a, T>]) -> ExResult<BinOpsWithReprs<'a, T>> {
    find_as_bin_op_with_reprs("^", ops)
}
fn minus_find_unary<'a, T: Copy + Debug>(
    ops: &[Operator<'a, T>],
) -> ExResult<UnaryOpWithReprs<'a, T>> {
    find_as_unary_op_with_reprs("-", ops)
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
                    let pow_op = pow_find(ops)?;
                    let log_op = find_as_unary_op_with_reprs("log", ops)?;
                    let mul_op = mul_find(ops)?;
                    let add_op = add_find(ops)?;
                    let sub_op = sub_find(ops)?;

                    let one = DeepEx::one();
                    let val = pow(f.val.clone(), g.val.clone(), pow_op.clone())?;
                    let g_minus_1 = g.val.clone().operate_bin(one, sub_op);
                    let der_1 = mul(
                        mul(
                            pow(f.val.clone(), g_minus_1, pow_op.clone())?,
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
                    Ok(one.with_only_unary_op(minus))
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
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
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
                        f.with_new_latest_unary_op(UnaryOpWithReprs::new()),
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
                    Ok(f.with_new_latest_unary_op(unary_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cos",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let mut sin = find_as_unary_op_with_reprs("sin", ops)?;
                    let minus = minus_find_unary(ops)?;
                    sin.append_after(minus);
                    Ok(f.with_new_latest_unary_op(sin))
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
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    let cos_squared_ex = f
                        .clone()
                        .with_new_latest_unary_op(cos_op)
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

                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    let inner_squared = f
                        .with_new_latest_unary_op(UnaryOpWithReprs::new())
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
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    let inner_squared = f
                        .with_new_latest_unary_op(UnaryOpWithReprs::new())
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
                    let pow_op = pow_find(ops)?;
                    let add_op = add_find(ops)?;
                    let div_op = div_find(ops)?;
                    let one = DeepEx::one();
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    let inner_squared = pow(
                        f.with_new_latest_unary_op(UnaryOpWithReprs::new()),
                        two,
                        pow_op,
                    )?;
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
                    Ok(f.with_new_latest_unary_op(cosh_op))
                },
            ),
        },
        PartialDerivative {
            repr: "cosh",
            bin_op: None,
            unary_outer_op: Some(
                |f: DeepEx<T>, ops: &[Operator<'a, T>]| -> ExResult<DeepEx<T>> {
                    let sinh_op = find_as_unary_op_with_reprs("sinh", ops)?;
                    Ok(f.with_new_latest_unary_op(sinh_op))
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
                    let two = DeepEx::from_num(T::from(2.0).unwrap());
                    sub(
                        one,
                        pow(f.with_new_latest_unary_op(tanh_op), two, pow_op)?,
                        sub_op,
                    )
                },
            ),
        },
    ]
}

#[cfg(test)]
use crate::{
    operators::VecOfUnaryFuncs, partial::details::prioritized_indices, util::assert_float_eq_f64,
    FlatEx, FloatOpsFactory,
};

#[cfg(test)]
pub fn parse<'a, T, F>(
    text: &'a str,
    ops: &[Operator<'a, T>],
    is_numeric: F,
) -> ExResult<DeepEx<'a, T>>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    F: Fn(&'a str) -> Option<&'a str>,
{
    let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
    parser::check_parsed_token_preconditions(&parsed_tokens)?;
    let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
    let (expr, _) =
        details::make_expression(&parsed_tokens[0..], &parsed_vars, UnaryOpWithReprs::new())?;
    Ok(expr)
}

#[test]
fn test_pmp() -> ExResult<()> {
    let x = 1.5f64;
    let fex = FlatEx::<f64>::from_str("+-+x")?;
    let deri = fex.partial(0)?;
    println!("{}", deri);
    let reference = -1.0;
    assert_float_eq_f64(deri.eval(&[x])?, reference);
    Ok(())
}

#[test]
fn test_sincosin() -> ExResult<()> {
    let x = 1.5f64;
    let fex = FlatEx::<f64>::from_str("sin(cos(sin(x)))")?;
    let deri = fex.partial(0)?;
    println!("{}", deri);
    let reference = x.cos() * (-x.sin().sin()) * x.sin().cos().cos();
    assert_float_eq_f64(deri.eval(&[x])?, reference);
    Ok(())
}

#[cfg(test)]
pub fn from_str(text: &str) -> ExResult<DeepEx<f64>> {
    let ops = FloatOpsFactory::<f64>::make();
    parse::<f64, _>(text, &ops, parser::is_numeric_text)
}

#[test]
fn test_reset_vars() {
    let deepex = from_str("2*z+x+y * .5").unwrap();
    let ref_vars = ["x", "y", "z"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex.var_names[i], *rv);
    }
    let deepex2 = from_str("a*c*b").unwrap();
    let ref_vars = ["a", "b", "c"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex2.var_names[i], *rv);
    }
    let (deepex_, deepex2_) = deepex.clone().var_names_union(deepex2.clone());
    let all_vars = ["a", "b", "c", "x", "y", "z"];
    for (i, av) in all_vars.iter().enumerate() {
        assert_eq!(deepex_.var_names[i], *av);
        assert_eq!(deepex2_.var_names[i], *av);
    }
    assert_eq!(deepex.unparse(), deepex_.unparse());
    assert_eq!(deepex2.unparse(), deepex2_.unparse());
}

#[test]
fn test_var_name_union() -> ExResult<()> {
    fn test(str_1: &str, str_2: &str, var_names: &[&str]) -> ExResult<()> {
        let first = from_str(str_1)?;
        let second = from_str(str_2)?;
        let (first, second) = first.var_names_union(second);

        assert_eq!(first.var_names.len(), var_names.len());
        assert_eq!(second.var_names.len(), var_names.len());
        for vn in first.var_names {
            assert!(var_names.contains(&vn));
        }
        for vn in second.var_names {
            assert!(var_names.contains(&vn));
        }
        Ok(())
    }

    test("x", "y", &["x", "y"])?;
    test("x+y*z", "z+y", &["x", "y", "z"])?;
    Ok(())
}

#[cfg(test)]
pub fn eval<T>(deepex: &DeepEx<T>, vars: &[T]) -> ExResult<T>
where
    T: Clone + Debug,
{
    let mut numbers = deepex
        .nodes
        .iter()
        .map(|node| -> ExResult<T> {
            match node {
                DeepNode::Num(n) => Ok(n.clone()),
                DeepNode::Var((idx, _)) => Ok(vars[*idx].clone()),
                DeepNode::Expr(e) => eval(e, vars),
            }
        })
        .collect::<ExResult<SmallVec<[T; N_NODES_ON_STACK]>>>()?;
    let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> =
        smallvec::smallvec![false; deepex.nodes.len()];
    let prio_indices = prioritized_indices(&deepex.bin_ops.ops, &deepex.nodes);
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
        numbers[num_idx - shift_left] = (deepex.bin_ops.ops[bin_op_idx].apply)(num_1, num_2);
        ignore[num_idx + shift_right] = true;
    }
    Ok(deepex.unary_op.op.apply(numbers[0].clone()))
}

#[test]
fn test_var_names() {
    let deepex = from_str("x+y+{x}+z*(-y)").unwrap();
    let reference: SmallVec<[&str; N_VARS_ON_STACK]> = smallvec::smallvec!["x", "y", "z"];
    assert_eq!(deepex.var_names, reference);
}

#[test]
fn test_deep_compile() {
    let ops = FloatOpsFactory::make();
    let nodes = vec![DeepNode::Num(4.5), DeepNode::Num(0.5), DeepNode::Num(1.4)];
    let bin_ops = BinOpsWithReprs {
        reprs: smallvec::smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec::smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec::smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec::smallvec![ops[6].unary().unwrap()]),
    };
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();

    let bin_ops = BinOpsWithReprs {
        reprs: smallvec::smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec::smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec::smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec::smallvec![ops[6].unary().unwrap()]),
    };
    let nodes = vec![
        DeepNode::Num(4.5),
        DeepNode::Num(0.5),
        DeepNode::Expr(Box::new(deep_ex)),
    ];
    let deepex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();
    assert_eq!(deepex.nodes.len(), 1);
    match deepex.nodes[0] {
        DeepNode::Num(n) => assert_float_eq_f64(deepex.unary_op.op.apply(n), n),
        _ => {
            unreachable!();
        }
    }
}

#[test]
fn test_deep_lift_node() {
    let deepex = from_str("(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))").unwrap();
    println!("{}", deepex);
    assert_eq!(
        format!("{}", deepex),
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))"
    );

    let deepex = from_str("(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "{a}+{x}^2.0*{x}^2.0");

    let deepex = from_str("1+(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "1.0+({a}+{x}^2.0*{x}^2.0)");
    let mut ddeepex = partial_deepex(1, deepex, &FloatOpsFactory::make()).unwrap();
    ddeepex.compile();
    println!("{}", ddeepex);
    assert_eq!(
        format!("{}", ddeepex),
        "(({x}^2.0)*({x}*2.0))+(({x}*2.0)*({x}^2.0))"
    );
}

#[test]
fn test_deep_compile_2() {
    let expr = from_str("1.0 * 3 * 2 * x / 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 2.0);
    let expr = from_str("x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+2+3+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))")
        .unwrap();
    assert_eq!(
        "{x}*0.25+{x}*8.0+5.0+7.0*sin({y})-{z}/sin(1.5/(1.0-{x}*4.0))",
        expr.unparse()
    );
    let expr = from_str("x + 1 - 2").unwrap();
    assert_float_eq_f64(eval(&expr, &[0.0]).unwrap(), -1.0);
    let expr = from_str("x - 1 + 2").unwrap();
    assert_float_eq_f64(eval(&expr, &[0.0]).unwrap(), 1.0);
    let expr = from_str("x * 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 4.0 / 3.0);
    let expr = from_str("x / 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 1.0 / 3.0);
}

#[test]
fn test_operate_unary() -> ExResult<()> {
    let lstr = "x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)";
    let deepex = from_str(lstr)?;
    let mut funcs = VecOfUnaryFuncs::new();
    funcs.push(|x: f64| x * 1.23456);
    let deepex = deepex.operate_unary(UnaryOpWithReprs {
        reprs: smallvec::smallvec!["eagle"],
        op: UnaryOp::from_vec(funcs),
    });
    assert_float_eq_f64(
        eval(&deepex, &[1.0, 1.75, 2.25])?,
        -0.23148000000000002 * 8.0,
    );
    Ok(())
}

#[test]
fn test_unparse() -> ExResult<()> {
    fn test(text: &str, text_ref: &str) -> ExResult<()> {
        let flatex = FlatEx::<f64>::from_str(text)?;
        assert_eq!(flatex.unparse(), text);
        let deepex = parse::<f64, _>(text, &FloatOpsFactory::make(), parser::is_numeric_text)?;
        assert_eq!(deepex.unparse(), text_ref);
        Ok(())
    }
    let text = "sin(5+var)^(1/{y})+{var}";
    let text_ref = "sin(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref)?;
    let text = "-(5+var)^(1/{y})+{var}";
    let text_ref = "-(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-(5+var)^(1/{y})))+{var}";
    let text_ref = "cos(sin(-(5.0+{var})^(1.0/{y})))+{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-5+var^(1/{y})))-{var}";
    let text_ref = "cos(sin(-5.0+{var}^(1.0/{y})))-{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-z+var*(1/{y})))+{var}";
    let text_ref = "cos(sin(-({z})+{var}*(1.0/{y})))+{var}";
    test(text, text_ref)?;
    Ok(())
}

#[test]
fn test_partial() {
    let ops = FloatOpsFactory::<f64>::make();
    let dut = from_str("z*sin(x)+cos(y)^(sin(z))").unwrap();
    let d_z = partial_deepex(2, dut.clone(), &ops).unwrap();
    assert_float_eq_f64(
        eval(
            &d_z,
            &[-0.18961918881278095, -6.383306547710852, 3.1742139703464503],
        )
        .unwrap(),
        -0.18346624475117082,
    );
    let dut = from_str("sin(x)/x^2").unwrap();
    let d_x = partial_deepex(0, dut, &ops).unwrap();
    assert_float_eq_f64(
        eval(&d_x, &[-0.18961918881278095]).unwrap(),
        -27.977974668662565,
    );

    let dut = from_str("x^y").unwrap();
    let d_x = partial_deepex(0, dut, &ops).unwrap();
    assert_float_eq_f64(eval(&d_x, &[7.5, 3.5]).unwrap(), 539.164392544148);
}

#[test]
fn test_partial_3_vars() {
    fn eval_(deepex: &DeepEx<f64>, vars: &[f64]) -> f64 {
        eval(&deepex, vars).unwrap()
    }
    fn assert(s: &str, vars: &[f64], ref_vals: &[f64]) {
        let ops = FloatOpsFactory::<f64>::make();
        let dut = from_str(s).unwrap();
        let d_x = partial_deepex(0, dut.clone(), &ops).unwrap();
        assert_float_eq_f64(eval_(&d_x, vars), ref_vals[0]);
        let d_y = partial_deepex(1, dut.clone(), &ops).unwrap();
        assert_float_eq_f64(eval_(&d_y, vars), ref_vals[1]);
        let d_z = partial_deepex(2, dut.clone(), &ops).unwrap();
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
    let ops = FloatOpsFactory::<f64>::make();
    let deepex = from_str("x * 2 * x").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = eval(&derivative, &[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = eval(&derivative, &[1.0]).unwrap();
    assert_float_eq_f64(result, 4.0);
}

#[test]
fn test_partial_cos_squared() {
    let ops = FloatOpsFactory::<f64>::make();
    let deepex = from_str("cos(y) ^ 2").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = eval(&derivative, &[0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = eval(&derivative, &[1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
}

#[test]
fn test_num_ops() {
    let ops = FloatOpsFactory::<f64>::make();
    let mul_op = find_as_bin_op_with_reprs("*", &ops).unwrap();
    fn eval_<'a>(deepex: &DeepEx<'a, f64>, vars: &[f64], val: f64) {
        assert_float_eq_f64(eval(&deepex, vars).unwrap(), val);
    }
    fn check_shape(deepex: &DeepEx<f64>, n_nodes: usize) {
        assert_eq!(deepex.nodes().len(), n_nodes);
        assert_eq!(deepex.bin_ops.ops.len(), n_nodes - 1);
        assert_eq!(deepex.bin_ops.reprs.len(), n_nodes - 1);
    }

    let minus_one = from_str("-1").unwrap();
    let one = mul(minus_one.clone(), minus_one.clone(), mul_op).unwrap();
    check_shape(&one, 1);
    eval_(&one, &[], 1.0);
}

#[test]
fn test_partial_combined() {
    let ops = FloatOpsFactory::<f64>::make();
    let deepex = from_str("sin(x) + cos(y) ^ 2").unwrap();
    let d_y = partial_deepex(1, deepex.clone(), &ops).unwrap();
    let result = eval(&d_y, &[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = eval(&d_y, &[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.9092974268256818);
    let d_x = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = eval(&d_x, &[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.5002954462477305);
    let result = eval(&d_x, &[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, 0.8438539587324921);
}

#[test]
fn test_partial_derivative_second_var() {
    let ops = FloatOpsFactory::<f64>::make();
    let deepex = from_str("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(1, deepex.clone(), &ops).unwrap();
    let result = eval(&derivative, &[231.431, 0.0]).unwrap();
    assert_float_eq_f64(result, 0.0);
    let result = eval(&derivative, &[-12.0, 1.0]).unwrap();
    assert_float_eq_f64(result, -0.8414709848078965);
}

#[test]
fn test_partial_derivative_first_var() {
    let ops = FloatOpsFactory::<f64>::make();

    let deepex = from_str("sin(x) + cos(y)").unwrap();
    let derivative = partial_deepex(0, deepex.clone(), &ops).unwrap();
    let result = eval(&derivative, &[0.0, 2345.03]).unwrap();
    assert_float_eq_f64(result, 1.0);
    let result = eval(&derivative, &[1.0, 43212.43]).unwrap();
    assert_float_eq_f64(result, 0.5403023058681398);
}

#[test]
fn test_partial_inner() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64], var_idx: usize) {
        let partial_derivative_ops = make_partial_derivative_ops::<f64>();
        let ops = FloatOpsFactory::<f64>::make();
        let deepex_1 = from_str(text).unwrap();
        let deri =
            partial_derivative_inner(var_idx, deepex_1, &partial_derivative_ops, &ops).unwrap();
        for i in 0..vals.len() {
            assert_float_eq_f64(eval(&deri, &[vals[i]]).unwrap(), ref_vals[i]);
        }
    }
    test("sin(x)", &[1.0, 0.0, 2.0], &[1.0, 1.0, 1.0], 0);
    test("sin(x^2)", &[1.0, 0.0, 2.0], &[2.0, 0.0, 4.0], 0);
}

#[test]
fn test_partial_outer() {
    fn test(text: &str, vals: &[f64], ref_vals: &[f64]) {
        let partial_derivative_ops = make_partial_derivative_ops::<f64>();
        let ops = FloatOpsFactory::<f64>::make();
        let deepex_1 = from_str(text).unwrap();
        let deepex = deepex_1.nodes()[0].clone();

        if let DeepNode::Expr(e) = deepex {
            let deri = partial_derivative_outer(*e, &partial_derivative_ops, &ops).unwrap();
            for i in 0..vals.len() {
                assert_float_eq_f64(eval(&deri, &[vals[i]]).unwrap(), ref_vals[i]);
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
    let ops = FloatOpsFactory::<f64>::make();

    let deepex = from_str("1")?;
    let derivative = partial_deepex(0, deepex, &ops)?;

    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 0.0),
        _ => unreachable!(),
    }
    let deepex = from_str("x")?;
    let derivative = partial_deepex(0, deepex, &ops)?;
    assert_eq!(derivative.nodes().len(), 1);
    assert_eq!(derivative.bin_ops().ops.len(), 0);
    match derivative.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(n, 1.0),
        _ => unreachable!(),
    }
    let deepex = from_str("x^2")?;
    let derivative = partial_deepex(0, deepex, &ops)?;
    let result = eval(&derivative, &[4.5])?;
    assert_float_eq_f64(result, 9.0);

    let deepex = from_str("sin(x)")?;
    let derivative = partial_deepex(0, deepex.clone(), &ops)?;
    let result = eval(&derivative, &[0.0])?;
    assert_float_eq_f64(result, 1.0);
    let result = eval(&derivative, &[1.0])?;
    assert_float_eq_f64(result, 0.5403023058681398);
    Ok(())
}
