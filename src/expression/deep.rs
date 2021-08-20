use crate::{
    definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK},
    expression::deep_details::{self, OverloadedOps, ADD_REPR, DIV_REPR, MUL_REPR, SUB_REPR},
    operators::{self, BinOp, UnaryOp},
    parser, ExParseError, Operator,
};
use num::Float;
use regex::Regex;
use smallvec::{smallvec, SmallVec};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    iter,
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
};

pub type ExprIdxVec = SmallVec<[usize; N_NODES_ON_STACK]>;

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;

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
    pub reprs: Vec<&'a str>,
    pub ops: BinOpVec<T>,
}
impl<'a, T: Copy> BinOpsWithReprs<'a, T> {
    pub fn new() -> BinOpsWithReprs<'a, T> {
        BinOpsWithReprs {
            reprs: vec![],
            ops: BinOpVec::new(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOpWithReprs<'a, T: Copy> {
    pub reprs: Vec<&'a str>,
    pub op: UnaryOp<T>,
}
impl<'a, T: Copy> UnaryOpWithReprs<'a, T> {
    pub fn new() -> UnaryOpWithReprs<'a, T> {
        UnaryOpWithReprs {
            reprs: vec![],
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
    overloaded_ops: Option<OverloadedOps<'a, T>>,
    var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
}

fn lift_nodes<'a, T: Copy + Debug>(deepex: &mut DeepEx<'a, T>) {
    if deepex.nodes.len() == 1 && deepex.unary_op.op.len() == 0 {
        match &deepex.nodes[0].clone() {
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
        for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (DeepNode::Num(num_1), DeepNode::Num(num_2)) = (node_1, node_2) {
                let bin_op_result = (self.bin_ops.ops[bin_op_idx].apply)(*num_1, *num_2);
                self.nodes[num_idx] = DeepNode::Num(bin_op_result);
                self.nodes.remove(num_idx + 1);
                // reduce indices after removed position
                for num_idx_after in num_inds.iter_mut() {
                    if *num_idx_after > num_idx {
                        *num_idx_after -= 1;
                    }
                }
                used_prio_indices.push(bin_op_idx);
            } else {
                break;
            }
        }

        let mut resulting_reprs = vec![];
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
    ) -> Result<DeepEx<'a, T>, ExParseError> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(ExParseError {
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
                overloaded_ops: None,
                var_names: found_vars,
            };
            expr.compile();
            Ok(expr)
        }
    }

    pub fn unparse(&self) -> String {
        let mut node_strings = self.nodes.iter().map(|n| match n {
            DeepNode::Num(n) => format!("{:?}", n),
            DeepNode::Var((_, var_name)) => format!("{{{}}}", var_name),
            DeepNode::Expr(e) => {
                if e.unary_op.op.len() == 0 {
                    format!("({})", e.unparse())
                } else {
                    e.unparse()
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

    pub fn from_node(node: DeepNode<'a, T>, overloaded_ops: OverloadedOps<'a, T>) -> DeepEx<'a, T> {
        let mut deepex =
            DeepEx::new(vec![node], BinOpsWithReprs::new(), UnaryOpWithReprs::new()).unwrap();
        deepex.set_overloaded_ops(Some(overloaded_ops));
        deepex
    }

    pub fn one(overloaded_ops: OverloadedOps<'a, T>) -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::one(), overloaded_ops)
    }

    pub fn one_like(other: &DeepEx<'a, T>) -> Result<DeepEx<'a, T>, ExParseError>
    where
        T: Float,
    {
        Ok(DeepEx::one(other.unpack_and_clone_overloaded_ops()?))
    }

    pub fn zero(overloaded_ops: OverloadedOps<'a, T>) -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::zero(), overloaded_ops)
    }

    pub fn with_new_unary_op(self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        Self {
            nodes: self.nodes,
            overloaded_ops: self.overloaded_ops,
            bin_ops: self.bin_ops,
            var_names: self.var_names,
            unary_op,
        }
    }

    pub fn from_str(text: &'a str) -> Result<DeepEx<'a, T>, ExParseError>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Float + FromStr,
    {
        let ops = operators::make_default_operators::<T>();
        DeepEx::from_ops(text, &ops)
    }

    pub fn from_ops(text: &'a str, ops: &[Operator<'a, T>]) -> Result<DeepEx<'a, T>, ExParseError>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
    {
        let parsed_tokens = parser::tokenize_and_analyze(text, ops, parser::is_numeric_text)?;
        let mut deepex = deep_details::parsed_tokens_to_deepex(&parsed_tokens)?;
        deepex.set_overloaded_ops(deep_details::find_overloaded_ops(ops));
        Ok(deepex)
    }

    pub fn from_pattern(
        text: &'a str,
        ops: &[Operator<'a, T>],
        number_regex_pattern: &str,
    ) -> Result<DeepEx<'a, T>, ExParseError>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
    {
        let beginning_number_regex_regex = format!("^({})", number_regex_pattern);
        let re_number = match Regex::new(beginning_number_regex_regex.as_str()) {
            Ok(regex) => regex,
            Err(_) => {
                return Err(ExParseError {
                    msg: "Cannot compile the passed number regex.".to_string(),
                })
            }
        };
        let is_numeric = |text: &'a str| parser::is_numeric_regex(&re_number, text);
        let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
        let mut deepex = deep_details::parsed_tokens_to_deepex(&parsed_tokens)?;
        deepex.set_overloaded_ops(deep_details::find_overloaded_ops(ops));
        Ok(deepex)
    }

    pub fn set_overloaded_ops(&mut self, ops: Option<OverloadedOps<'a, T>>) {
        self.overloaded_ops = ops;
    }

    pub fn bin_ops(&self) -> &BinOpsWithReprs<T> {
        &self.bin_ops
    }

    pub fn unary_op(&self) -> &UnaryOpWithReprs<T> {
        &self.unary_op
    }

    pub fn nodes(&self) -> &Vec<DeepNode<'a, T>> {
        &self.nodes
    }

    pub fn unpack_and_clone_overloaded_ops(&self) -> Result<OverloadedOps<'a, T>, ExParseError> {
        self.overloaded_ops.clone().ok_or(ExParseError {
            msg: "cannot unpack overloaded ops when there are none".to_string(),
        })
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
        let overloaded_ops = self.overloaded_ops.clone();

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
        resex.overloaded_ops = overloaded_ops;
        resex.compile();
        resex
    }

    /// Applies a unary operator to self
    pub fn operate_unary(mut self, mut unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.append_front(&mut unary_op);
        self.compile();
        self
    }

    /// Applies one of the binary overloaded operators to self and other.
    ///
    /// # Panics
    ///
    /// if an overloaded operator has not been defined
    ///
    pub fn operate_overloaded(self, other: Self, repr: &'a str) -> Self {
        if self.overloaded_ops.is_none() {
            panic!("overloaded operators not available");
        }
        let overloaded_ops = self.overloaded_ops.clone();
        let op = overloaded_ops.clone().unwrap().by_repr(repr);

        let ops = smallvec![op.bin_op.unwrap()];

        let bin_op = BinOpsWithReprs {
            reprs: vec![repr],
            ops,
        };
        self.operate_bin(other, bin_op)
    }
}

impl<'a, T: Copy + Debug> Add for DeepEx<'a, T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        self.operate_overloaded(other, ADD_REPR)
    }
}

impl<'a, T: Copy + Debug> Sub for DeepEx<'a, T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        self.operate_overloaded(other, SUB_REPR)
    }
}

impl<'a, T: Copy + Debug> Mul for DeepEx<'a, T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        self.operate_overloaded(other, MUL_REPR)
    }
}

impl<'a, T: Copy + Debug> Div for DeepEx<'a, T> {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self.operate_overloaded(other, DIV_REPR)
    }
}

impl<'a, T: Copy + Debug> Display for DeepEx<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.unparse())
    }
}

#[cfg(test)]
use {
    super::flat::flatten,
    crate::{
        expression::partial_derivatives::partial_deepex,
        operators::make_default_operators,
        util::{assert_float_eq, assert_float_eq_f64},
    },
    rand::{thread_rng, Rng},
    std::ops::Range,
};

#[test]
fn test_reset_vars() {
    let deepex = DeepEx::<f64>::from_str("2*z+x+y * .5").unwrap();
    let ref_vars = ["x", "y", "z"];
    for i in 0..ref_vars.len() {
        assert_eq!(deepex.var_names[i], ref_vars[i]);
    }
    let deepex2 = DeepEx::<f64>::from_str("a*c*b").unwrap();
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
    assert_eq!(deepex.unparse(), deepex_.unparse());
    assert_eq!(deepex2.unparse(), deepex2_.unparse());
    let flatex = flatten(deepex);
    let flatex2 = flatten(deepex2);
    let flatex_ = flatten(deepex_);
    let flatex2_ = flatten(deepex2_);
    assert_float_eq_f64(flatex.eval(&[2.0, 6.0, 1.5]).unwrap(), 8.0);
    assert_float_eq_f64(flatex2.eval(&[3.0, 5.0, 4.0]).unwrap(), 60.0);
    assert_float_eq_f64(flatex_.eval(&[3.0, 5.0, 4.0, 2.0, 6.0, 1.5]).unwrap(), 8.0);
    assert_float_eq_f64(
        flatex2_.eval(&[3.0, 5.0, 4.0, 2.0, 6.0, 1.5]).unwrap(),
        60.0,
    );
}

#[test]
fn test_var_name_union() {
    fn from_str(text: &str) -> DeepEx<f64> {
        DeepEx::from_str(text).unwrap()
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
fn test_operator_overloading() {
    fn from_str(text: &str) -> DeepEx<f64> {
        DeepEx::from_str(text).unwrap()
    }
    fn eval<'a>(deepex: &DeepEx<'a, f64>, vars: &[f64], val: f64) {
        assert_float_eq_f64(flatten(deepex.clone()).eval(vars).unwrap(), val);
    }

    fn check_shape<'a>(deepex: &DeepEx<'a, f64>, n_nodes: usize) {
        assert_eq!(deepex.nodes.len(), n_nodes);
        assert_eq!(deepex.bin_ops.ops.len(), n_nodes - 1);
        assert_eq!(deepex.bin_ops.reprs.len(), n_nodes - 1);
    }

    let one = from_str("1");
    let two = one.clone() + one.clone();
    check_shape(&two, 1);
    eval(&two, &[], 2.0);
    
    let minus_one = from_str("-1");
    let one = minus_one.clone() * minus_one.clone();
    check_shape(&one, 1);
    eval(&one, &[], 1.0);

    let x_squared = from_str("x*x");
    check_shape(&x_squared, 2);
    let two_x_squared = two.clone() * x_squared.clone();
    check_shape(&two_x_squared, 2);
    eval(&two_x_squared, &[0.0], 0.0);
    eval(&two_x_squared, &[1.0], 2.0);
    eval(&two_x_squared, &[2.0], 8.0);
    eval(&two_x_squared, &[3.0], 18.0);
    let some_expr = from_str("x") + from_str("x") * from_str("2") / from_str("x^(.5)");
    check_shape(&some_expr, 2);
    eval(&some_expr, &[4.0], 8.0);

    let x_plus_y_plus_z = from_str("x+y+z");
    check_shape(&x_plus_y_plus_z, 3);
    let y_minus_z = from_str("y-z");
    check_shape(&y_minus_z, 2);
    let prod_of_above = x_plus_y_plus_z.clone() * y_minus_z.clone();
    check_shape(&prod_of_above, 2);
    eval(&prod_of_above, &[1.0, 4.0, 8.0], -52.0);
    let div_of_above = x_plus_y_plus_z.clone() / y_minus_z.clone();
    eval(&div_of_above, &[1.0, 4.0, 8.0], -3.25);
    let sub_of_above = x_plus_y_plus_z.clone() - y_minus_z.clone();
    eval(&sub_of_above, &[1.0, 4.0, 8.0], 17.0);
    let add_of_above = x_plus_y_plus_z + y_minus_z.clone();
    eval(&add_of_above, &[1.0, 4.0, 8.0], 9.0);
    let x_plus_cossin_y_plus_z = from_str("x+cos(sin(y+z))");
    let prod_of_above = x_plus_cossin_y_plus_z * y_minus_z;
    eval(&prod_of_above, &[1.0, 4.0, 8.0], -7.4378625090980925);
}

#[test]
fn test_partial_finite() {
    let ops = make_default_operators::<f64>();
    fn test<'a>(sut: &str, ops: &'a [Operator<'a, f64>], range: Range<f64>) {
        let dut = DeepEx::<f64>::from_str(sut).unwrap();
        let n_vars = dut.n_vars();
        let step = 1e-5;
        let mut rng = thread_rng();

        let x0s: Vec<f64> = (0..n_vars).map(|_| rng.gen_range(range.clone())).collect();
        println!(
            "test_partial_finite - checking derivatives at {:?} for {}",
            x0s, sut
        );
        println!("test_partial_finite - dut vars {:?}", dut.var_names);
        for (var_idx, var_name) in dut.var_names.iter().enumerate() {
            let x1s: Vec<f64> = x0s
                .iter()
                .enumerate()
                .map(|(i, x0)| if i == var_idx { x0 + step } else { *x0 })
                .collect();
            let flat_dut = flatten(dut.clone());
            let f0 = flat_dut.eval(&x0s).unwrap();
            let f1 = flat_dut.eval(&x1s).unwrap();
            let finite_diff = (f1 - f0) / step;
            let deri = partial_deepex(var_idx, dut.clone(), &ops).unwrap();
            let flat_deri = flatten(deri.clone()).eval(&x0s).unwrap();
            println!(
                "test_partial_finite -\n {} (derivative)\n {} (finite diff)",
                flat_deri, finite_diff
            );
            let msg = format!("sut {}, d_{} is {}", sut, var_name, deri);
            println!("test_partial_finite - {}", msg);
            assert_float_eq::<f64>(flat_deri, finite_diff, 1e-5, 1e-3, msg.as_str());
        }
    }

    test("z*sin(x)+cos(y)^(1 + x^2)/(sin(z))", &ops, 0.01..1.0);
    test("sin(sin(x+z))", &ops, -10.0..10.0);
    test("x^x", &ops, 0.01..2.0);
    test("x^y", &ops, 4.036286084344371..4.036286084344372);
    test("z+sin(x)+cos(y)", &ops, -1.0..1.0);
    test("sin(cos(sin(z)))", &ops, -10.0..10.0);
    test("sin(x+z)", &ops, -10.0..10.0);
    test("sin(x-z)", &ops, -10.0..10.0);
    test("y-sin(x-z)", &ops, -10.0..10.0);
    test("(sin(x)^2)/x/4", &ops, -10.0..10.0);
    test("1/x", &ops, -10.0..10.0);
    test("sin(y+x)/((x*2)/y)*(2*x)", &ops, -1.0..1.0);
    test("log(x^2)", &ops, 0.1..10.0);
    test("tan(x)", &ops, -1.0..1.0);
    test("tan(exp(x))", &ops, -1000.0..0.0);
    test("exp(y-x)", &ops, -1.0..1.0);
    test("sqrt(exp(y-x))", &ops, -1000.0..0.0);
    test("sqrt(x)", &ops, 0.0..10000.0);
    test("asin(x)", &ops, -1.0..1.0);
    test("acos(x)", &ops, -1.0..1.0);
    test("atan(x)", &ops, -1.0..1.0);
    test("asin(sqrt(x+y))", &ops, 0.0..0.5);
    test("sinh(x)", &ops, -1.0..1.0);
    test("cosh(x)", &ops, -1.0..1.0);
    test("tanh(x)", &ops, -1.0..1.0);
}

#[test]
fn test_var_names() {
    let deepex = DeepEx::<f64>::from_str("x+y+{x}+z*(-y)").unwrap();
    let reference: SmallVec<[&str; N_VARS_ON_STACK]> = smallvec!["x", "y", "z"];
    assert_eq!(deepex.var_names, reference);
}

#[test]
fn test_deep_compile() {
    let ops = make_default_operators();
    let nodes = vec![DeepNode::Num(4.5), DeepNode::Num(0.5), DeepNode::Num(1.4)];
    let bin_ops = BinOpsWithReprs {
        reprs: vec![ops[1].repr, ops[3].repr],
        ops: smallvec![ops[1].bin_op.unwrap(), ops[3].bin_op.unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: vec![ops[6].repr],
        op: UnaryOp::from_vec(smallvec![ops[6].unary_op.unwrap()]),
    };
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();

    let bin_ops = BinOpsWithReprs {
        reprs: vec![ops[1].repr, ops[3].repr],
        ops: smallvec![ops[1].bin_op.unwrap(), ops[3].bin_op.unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: vec![ops[6].repr],
        op: UnaryOp::from_vec(smallvec![ops[6].unary_op.unwrap()]),
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
        DeepEx::<f64>::from_str("(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))").unwrap();
    println!("{}", deepex);
    assert_eq!(
        format!("{}", deepex),
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))"
    );

    let deepex = DeepEx::<f64>::from_str("(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "{a}+{x}^2.0*{x}^2.0");

    let deepex = DeepEx::<f64>::from_str("1+(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "1.0+({a}+{x}^2.0*{x}^2.0)");
    let mut ddeepex = partial_deepex(1, deepex, &make_default_operators()).unwrap();
    ddeepex.compile();
    println!("{}", ddeepex);
    assert_eq!(
        format!("{}", ddeepex),
        "(({x}^2.0)*({x}*2.0))+(({x}*2.0)*({x}^2.0))"
    );
}
