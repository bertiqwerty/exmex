use crate::{
    definitions::{
        N_BINOPS_OF_DEEPEX_ON_STACK, N_NODES_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK,
        N_VARS_ON_STACK,
    },
    expression::deep_details::{
        self, prioritized_indices, BinOpsWithReprsBuf, UnaryOpWithReprsBuf,
    },
    operators::{BinOp, DefaultOpsFactory, MakeOperators, UnaryOp},
    parser, ExError, ExResult, Operator,
};
use num::Float;
use regex::Regex;
use smallvec::{smallvec, SmallVec};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    iter,
    str::FromStr,
};

pub type ExprIdxVec = SmallVec<[usize; N_NODES_ON_STACK]>;

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;

pub fn parse<'a, T, F>(
    text: &'a str,
    ops: &[Operator<'a, T>],
    is_numeric: F,
) -> ExResult<DeepEx<'a, T>>
where
    T: Copy + Debug + FromStr,
    <T as FromStr>::Err: Debug,
    F: Fn(&'a str) -> Option<&'a str>,
{
    let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
    parser::check_parsed_token_preconditions(&parsed_tokens)?;
    let parsed_vars = deep_details::find_parsed_vars(&parsed_tokens);
    let (expr, _) =
        deep_details::make_expression(&parsed_tokens[0..], &parsed_vars, UnaryOpWithReprs::new())?;
    Ok(expr)
}

/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DeepNode<'a, T: Copy + Debug> {
    Expr(DeepEx<'a, T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var((usize, &'a str)),
}
impl<'a, T: Copy + Debug> DeepNode<'a, T>
where
    T: Float,
{
    pub fn zero() -> Self {
        DeepNode::Num(T::from(0.0).unwrap())
    }
    pub fn one() -> Self {
        DeepNode::Num(T::from(1.0).unwrap())
    }
    pub fn num(n: T) -> Self {
        DeepNode::Num(n)
    }
}
impl<'a, T: Copy + Debug> Debug for DeepNode<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DeepNode::Expr(e) => write!(f, "{}", e),
            DeepNode::Num(n) => write!(f, "{:?}", n),
            DeepNode::Var((_, var_name)) => write!(f, "{}", var_name),
        }
    }
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprs<'a, T: Copy> {
    pub reprs: SmallVec<[&'a str; N_BINOPS_OF_DEEPEX_ON_STACK]>,
    pub ops: BinOpVec<T>,
}
impl<'a, T: Copy> BinOpsWithReprs<'a, T> {
    pub fn new() -> Self {
        BinOpsWithReprs {
            reprs: smallvec![],
            ops: BinOpVec::new(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOpWithReprs<'a, T: Copy> {
    pub reprs: SmallVec<[&'a str; N_UNARYOPS_OF_DEEPEX_ON_STACK]>,
    pub op: UnaryOp<T>,
}
impl<'a, T: Copy> UnaryOpWithReprs<'a, T> {
    pub fn new() -> UnaryOpWithReprs<'a, T> {
        UnaryOpWithReprs {
            reprs: smallvec![],
            op: UnaryOp::new(),
        }
    }

    pub fn append_front(&mut self, other: &mut UnaryOpWithReprs<'a, T>) {
        self.op.append_front(&mut other.op);
        self.reprs = other
            .reprs
            .iter()
            .chain(self.reprs.iter())
            .copied()
            .collect();
    }
}

/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T: Copy + Debug> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T>>,
    /// Binary operators applied to the nodes according to their priority.
    pub bin_ops: BinOpsWithReprs<'a, T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: UnaryOpWithReprs<'a, T>,
    var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
}

fn lift_nodes<'a, T: Copy + Debug>(deepex: &mut DeepEx<'a, T>) {
    if deepex.nodes.len() == 1 && deepex.unary_op.op.len() == 0 {
        match &deepex.nodes[0] {
            DeepNode::Expr(e) => {
                *deepex = e.clone();
            }
            _ => (),
        }
    } else {
        for node in &mut deepex.nodes {
            if let DeepNode::Expr(e) = node {
                if e.nodes.len() == 1 && e.unary_op.op.len() == 0 {
                    match &mut e.nodes[0] {
                        DeepNode::Num(n) => *node = DeepNode::Num(*n),
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

impl<'a, T: Copy + Debug> DeepEx<'a, T> {
    /// Evaluates all operators with numbers as operands.
    pub fn compile(&mut self) {
        lift_nodes(self);

        let prio_indices = deep_details::prioritized_indices(&self.bin_ops.ops, &self.nodes);
        let mut num_inds = prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();

        let mut already_declined: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec![false; self.nodes.len()];

        for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (DeepNode::Num(num_1), DeepNode::Num(num_2)) = (node_1, node_2) {
                if !(already_declined[num_idx] || already_declined[num_idx + 1]) {
                    let bin_op_result = (self.bin_ops.ops[bin_op_idx].apply)(*num_1, *num_2);
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

        let mut resulting_reprs = smallvec![];
        self.bin_ops.ops = self
            .bin_ops
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|(i, bin_op)| {
                resulting_reprs.push(self.bin_ops.reprs[i]);
                *bin_op
            })
            .collect();
        self.bin_ops.reprs = resulting_reprs;

        if self.nodes.len() == 1 {
            if let DeepNode::Num(n) = self.nodes[0] {
                self.nodes[0] = DeepNode::Num(self.unary_op.op.apply(n));
                self.unary_op.op.clear();
                self.unary_op.reprs.clear();
            }
        }
    }

    pub fn n_vars(&self) -> usize {
        self.var_names.len()
    }

    pub fn new(
        nodes: Vec<DeepNode<'a, T>>,
        bin_ops: BinOpsWithReprs<'a, T>,
        unary_op: UnaryOpWithReprs<'a, T>,
    ) -> ExResult<DeepEx<'a, T>> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(ExError {
                msg: format!(
                    "mismatch between number of nodes {:?} and binary operators {:?} ({} vs {})",
                    nodes,
                    bin_ops.ops,
                    nodes.len(),
                    bin_ops.ops.len()
                ),
            })
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

    pub fn from_node(node: DeepNode<'a, T>) -> DeepEx<'a, T> {
        DeepEx::new(vec![node], BinOpsWithReprs::new(), UnaryOpWithReprs::new()).unwrap()
    }

    pub fn one() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::one())
    }

    pub fn zero() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::zero())
    }

    pub fn from_num(x: T) -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::num(x))
    }

    pub fn with_new_unary_op(self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        Self {
            nodes: self.nodes,
            bin_ops: self.bin_ops,
            var_names: self.var_names,
            unary_op,
        }
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
        self.nodes.len() == 1
            && match &self.nodes[0] {
                DeepNode::Num(n) => self.unary_op.op.apply(*n) == num,
                DeepNode::Expr(e) => e.is_num(num),
                _ => false,
            }
    }

    pub fn is_one(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(1.0).unwrap())
    }

    pub fn is_zero(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(0.0).unwrap())
    }

    pub fn var_names_union(self, other: Self) -> (Self, Self) {
        fn reset_vars<'a, T: Copy + Debug>(
            deepex: &mut DeepEx<'a, T>,
            new_var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
        ) {
            for node in &mut deepex.nodes {
                match node {
                    DeepNode::Expr(e) => reset_vars(e, new_var_names.clone()),
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
            deepex.var_names = new_var_names;
        }
        let mut all_var_names = self.var_names.clone();
        for name in other.var_names.clone() {
            if !all_var_names.contains(&name) {
                all_var_names.push(name);
            }
        }
        all_var_names.sort_unstable();
        let mut self_vars_updated = self;
        let mut other_vars_updated = other;
        reset_vars(&mut self_vars_updated, all_var_names.clone());
        reset_vars(&mut other_vars_updated, all_var_names);
        (self_vars_updated, other_vars_updated)
    }

    pub fn var_names_like_other(mut self, other: &Self) -> Self {
        self.var_names = other.var_names.clone();
        self
    }

    /// Applies a binary operator to self and other
    pub fn operate_bin(self, other: Self, bin_op: BinOpsWithReprs<'a, T>) -> Self {
        let (self_vars_updated, other_vars_updated) = self.var_names_union(other);
        let mut resex = DeepEx::new(
            vec![
                DeepNode::Expr(self_vars_updated),
                DeepNode::Expr(other_vars_updated),
            ],
            bin_op,
            UnaryOpWithReprs::new(),
        )
        .unwrap();
        resex.compile();
        resex
    }

    /// Applies a unary operator to self
    pub fn operate_unary(mut self, mut unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.append_front(&mut unary_op);
        self.compile();
        self
    }

    pub fn unparse_raw(&self) -> String {
        let mut node_strings = self.nodes.iter().map(|n| match n {
            DeepNode::Num(n) => format!("{:?}", n),
            DeepNode::Var((_, var_name)) => format!("{{{}}}", var_name),
            DeepNode::Expr(e) => {
                if e.unary_op.op.len() == 0 {
                    format!("({})", e.unparse_raw())
                } else {
                    e.unparse_raw()
                }
            }
        });
        let mut bin_op_strings = self.bin_ops.reprs.iter();
        // a valid expression has at least one node
        let first_node_str = node_strings.next().unwrap();
        let node_with_bin_ops_string = node_strings.fold(first_node_str, |mut res, node_str| {
            let bin_op_str = bin_op_strings.next().unwrap();
            res.push_str(bin_op_str);
            res.push_str(node_str.as_str());
            res
        });
        let unary_op_string = self
            .unary_op
            .reprs
            .iter()
            .fold(String::new(), |mut res, uop_str| {
                res.push_str(uop_str);
                res.push('(');
                res
            });
        let closings = iter::repeat(")").take(self.unary_op.op.len()).fold(
            String::new(),
            |mut res, closing| {
                res.push_str(closing);
                res
            },
        );
        if self.unary_op.op.len() == 0 {
            node_with_bin_ops_string
        } else {
            format!(
                "{}{}{}",
                unary_op_string, node_with_bin_ops_string, closings
            )
        }
    }

    pub fn from_str_float(text: &'a str) -> ExResult<DeepEx<'a, T>>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Float + FromStr,
    {
        let ops = DefaultOpsFactory::<T>::make();
        DeepEx::from_ops(text, &ops)
    }

    pub fn from_ops(text: &'a str, ops: &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
    {
        parse(text, ops, parser::is_numeric_text)
    }

    pub fn from_pattern(
        text: &'a str,
        ops: &[Operator<'a, T>],
        number_regex_pattern: &str,
    ) -> ExResult<DeepEx<'a, T>>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
    {
        let beginning_num_re_pattern = format!("^({})", number_regex_pattern);
        let re_number = match Regex::new(beginning_num_re_pattern.as_str()) {
            Ok(regex) => regex,
            Err(_) => {
                return Err(ExError {
                    msg: "Cannot compile the passed number regex.".to_string(),
                })
            }
        };
        let is_numeric = |text: &'a str| parser::is_numeric_regex(&re_number, text);
        parse(text, ops, is_numeric)
    }

    pub fn eval(&self, vars: &[T]) -> ExResult<T> {
        let mut numbers = self
            .nodes
            .iter()
            .map(|node| -> ExResult<T> {
                match node {
                    DeepNode::Num(n) => Ok(*n),
                    DeepNode::Var((idx, _)) => Ok(vars[*idx]),
                    DeepNode::Expr(e) => e.eval(vars),
                }
            })
            .collect::<ExResult<SmallVec<[T; N_NODES_ON_STACK]>>>()?;
        let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> = smallvec![false; self.nodes.len()];
        let prio_indices = prioritized_indices(&self.bin_ops.ops, &self.nodes);
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
            let num_1 = numbers[num_idx - shift_left];
            let num_2 = numbers[num_idx + shift_right];
            numbers[num_idx - shift_left] = (self.bin_ops.ops[bin_op_idx].apply)(num_1, num_2);
            ignore[num_idx + shift_right] = true;
        }
        Ok(self.unary_op.op.apply(numbers[0]))
    }
}

impl<'a, T: Copy + Debug> Display for DeepEx<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.unparse_raw())
    }
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum DeepBufNode<T: Copy + Debug> {
    Expr(DeepBuf<T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var((usize, String)),
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepBuf<T: Copy + Debug> {
    pub nodes: Vec<DeepBufNode<T>>,
    /// Binary operators applied to the nodes according to their priority.
    pub bin_ops: BinOpsWithReprsBuf<T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    pub unary_op: UnaryOpWithReprsBuf<T>,
    pub unparsed: String,
    pub var_names: SmallVec<[String; N_VARS_ON_STACK]>,
}

impl<'a, T: Copy + Debug> DeepBuf<T> {
    pub fn from_deepex(deepex: &DeepEx<'a, T>) -> Self {
        Self {
            nodes: deepex
                .nodes()
                .iter()
                .map(|node| match node {
                    DeepNode::Expr(e) => DeepBufNode::Expr(Self::from_deepex(e)),
                    DeepNode::Num(n) => DeepBufNode::Num(*n),
                    DeepNode::Var(v) => DeepBufNode::Var((v.0, v.1.to_string())),
                })
                .collect(),
            bin_ops: BinOpsWithReprsBuf::from_deepex(deepex.bin_ops()),
            unary_op: UnaryOpWithReprsBuf::from_deepex(deepex.unary_op()),
            unparsed: deepex.unparse_raw(),
            var_names: deepex.var_names.iter().map(|vn| vn.to_string()).collect(),
        }
    }
    pub fn to_deepex(&'a self, ops: &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>> {
        let mut deepex = DeepEx::new(
            self.nodes
                .iter()
                .map(|node| -> ExResult<_> {
                    match node {
                        DeepBufNode::Expr(e) => Ok(DeepNode::Expr(e.to_deepex(ops)?)),
                        DeepBufNode::Num(n) => Ok(DeepNode::Num(*n)),
                        DeepBufNode::Var(v) => Ok(DeepNode::Var((v.0, v.1.as_str()))),
                    }
                })
                .collect::<ExResult<_>>()?,
            self.bin_ops.to_deepex(),
            self.unary_op.to_deepex(),
        )?;
        deepex.var_names = self.var_names.iter().map(|vn| vn.as_str()).collect();
        Ok(deepex)
    }
}

#[cfg(test)]
use {
    crate::{
        expression::partial_derivatives::partial_deepex,
        util::{assert_float_eq, assert_float_eq_f64},
    },
    rand::{thread_rng, Rng},
    std::ops::Range,
};

#[test]
fn test_reset_vars() {
    let deepex = DeepEx::<f64>::from_str_float("2*z+x+y * .5").unwrap();
    let ref_vars = ["x", "y", "z"];
    for i in 0..ref_vars.len() {
        assert_eq!(deepex.var_names[i], ref_vars[i]);
    }
    let deepex2 = DeepEx::<f64>::from_str_float("a*c*b").unwrap();
    let ref_vars = ["a", "b", "c"];
    for i in 0..ref_vars.len() {
        assert_eq!(deepex2.var_names[i], ref_vars[i]);
    }
    let (deepex_, deepex2_) = deepex.clone().var_names_union(deepex2.clone());
    let all_vars = ["a", "b", "c", "x", "y", "z"];
    for i in 0..all_vars.len() {
        assert_eq!(deepex_.var_names[i], all_vars[i]);
        assert_eq!(deepex2_.var_names[i], all_vars[i]);
    }
    assert_eq!(deepex.unparse_raw(), deepex_.unparse_raw());
    assert_eq!(deepex2.unparse_raw(), deepex2_.unparse_raw());
    assert_float_eq_f64(deepex.eval(&[2.0, 6.0, 1.5]).unwrap(), 8.0);
    assert_float_eq_f64(deepex2.eval(&[3.0, 5.0, 4.0]).unwrap(), 60.0);
    assert_float_eq_f64(deepex_.eval(&[3.0, 5.0, 4.0, 2.0, 6.0, 1.5]).unwrap(), 8.0);
    assert_float_eq_f64(
        deepex2_.eval(&[3.0, 5.0, 4.0, 2.0, 6.0, 1.5]).unwrap(),
        60.0,
    );
}

#[test]
fn test_var_name_union() {
    fn from_str(text: &str) -> DeepEx<f64> {
        DeepEx::from_str_float(text).unwrap()
    }
    fn test(str_1: &str, str_2: &str, var_names: &[&str]) {
        let first = from_str(str_1);
        let second = from_str(str_2);
        let (first, second) = first.var_names_union(second);

        assert_eq!(first.n_vars(), var_names.len());
        assert_eq!(second.n_vars(), var_names.len());
        for vn in first.var_names {
            assert!(var_names.contains(&vn));
        }
        for vn in second.var_names {
            assert!(var_names.contains(&vn));
        }
    }

    test("x", "y", &vec!["x", "y"]);
    test("x+y*z", "z+y", &vec!["x", "y", "z"]);
}

#[test]
fn test_partial_finite() {
    let ops = DefaultOpsFactory::<f64>::make();
    fn test<'a>(sut: &str, ops: &'a [Operator<'a, f64>], range: Range<f64>) {
        let dut = DeepEx::<f64>::from_str_float(sut).unwrap();
        let n_vars = dut.n_vars();
        let step = 1e-5;
        let mut rng = thread_rng();

        let x0s: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(range.clone())).collect();
        println!(
            "test_partial_finite - checking derivatives at {:?} for {}",
            x0s, sut
        );
        for (var_idx, var_name) in dut.var_names.iter().enumerate() {
            let x1s: Vec<f64> = x0s
                .iter()
                .enumerate()
                .map(|(i, x0)| if i == var_idx { x0 + step } else { *x0 })
                .collect();

            let f0 = dut.eval(&x0s).unwrap();
            let f1 = dut.eval(&x1s).unwrap();
            let finite_diff = (f1 - f0) / step;
            let deri = partial_deepex(var_idx, dut.clone(), &ops).unwrap();
            let deri = deri.eval(&x0s).unwrap();
            println!(
                "test_partial_finite -\n {} (derivative)\n {} (finite diff)",
                deri, finite_diff
            );
            let msg = format!("sut {}, d_{} is {}", sut, var_name, deri);
            println!("test_partial_finite - {}", msg);
            assert_float_eq::<f64>(deri, finite_diff, 1e-5, 1e-3, msg.as_str());
        }
    }
    test("sqrt(x)", &ops, 0.0..10000.0);
    test("asin(x)", &ops, -1.0..1.0);
    test("acos(x)", &ops, -1.0..1.0);
    test("atan(x)", &ops, -1.0..1.0);
    test("1/x", &ops, -10.0..10.0);
    test("x^x", &ops, 0.01..2.0);
    test("x^y", &ops, 4.036286084344371..4.036286084344372);
    test("z+sin(x)+cos(y)", &ops, -1.0..1.0);
    test("sin(cos(sin(z)))", &ops, -10.0..10.0);
    test("sin(x+z)", &ops, -10.0..10.0);
    test("sin(x-z)", &ops, -10.0..10.0);
    test("y-sin(x-z)", &ops, -10.0..10.0);
    test("(sin(x)^2)/x/4", &ops, -10.0..10.0);
    test("sin(y+x)/((x*2)/y)*(2*x)", &ops, -1.0..1.0);
    test("z*sin(x)+cos(y)^(1 + x^2)/(sin(z))", &ops, 0.01..1.0);
    test("log(x^2)", &ops, 0.1..10.0);
    test("tan(x)", &ops, -1.0..1.0);
    test("tan(exp(x))", &ops, -1000.0..0.0);
    test("exp(y-x)", &ops, -1.0..1.0);
    test("sqrt(exp(y-x))", &ops, -1000.0..0.0);
    test("sin(sin(x+z))", &ops, -10.0..10.0);
    test("asin(sqrt(x+y))", &ops, 0.0..0.5);
}

#[test]
fn test_var_names() {
    let deepex = DeepEx::<f64>::from_str_float("x+y+{x}+z*(-y)").unwrap();
    let reference: SmallVec<[&str; N_VARS_ON_STACK]> = smallvec!["x", "y", "z"];
    assert_eq!(deepex.var_names, reference);
}

#[test]
fn test_deep_compile() {
    let ops = DefaultOpsFactory::make();
    let nodes = vec![DeepNode::Num(4.5), DeepNode::Num(0.5), DeepNode::Num(1.4)];
    let bin_ops = BinOpsWithReprs {
        reprs: smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec![ops[6].unary().unwrap()]),
    };
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();

    let bin_ops = BinOpsWithReprs {
        reprs: smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec![ops[6].unary().unwrap()]),
    };
    let nodes = vec![
        DeepNode::Num(4.5),
        DeepNode::Num(0.5),
        DeepNode::Expr(deep_ex),
    ];
    let deepex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();
    assert_eq!(deepex.nodes.len(), 1);
    match deepex.nodes[0] {
        DeepNode::Num(n) => assert_eq!(deepex.unary_op.op.apply(n), n),
        _ => {
            assert!(false);
        }
    }
}

#[test]
fn test_deep_compile_2() {
    let deepex =
        DeepEx::<f64>::from_str_float("(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))")
            .unwrap();
    println!("{}", deepex);
    assert_eq!(
        format!("{}", deepex),
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))"
    );

    let deepex = DeepEx::<f64>::from_str_float("(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "{a}+{x}^2.0*{x}^2.0");

    let deepex = DeepEx::<f64>::from_str_float("1+(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "1.0+({a}+{x}^2.0*{x}^2.0)");
    let mut ddeepex = partial_deepex(1, deepex, &DefaultOpsFactory::make()).unwrap();
    ddeepex.compile();
    println!("{}", ddeepex);
    assert_eq!(
        format!("{}", ddeepex),
        "(({x}^2.0)*({x}*2.0))+(({x}*2.0)*({x}^2.0))"
    );
}
