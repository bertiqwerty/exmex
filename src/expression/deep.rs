use super::deep_details::{
    self, find_overloaded_ops, OverloadedOps, ADD_REPR, DIV_REPR, MUL_REPR, SUB_REPR,
};
use crate::definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK};
use crate::operators::VecOfUnaryFuncs;
use crate::{
    operators,
    operators::{BinOp, UnaryOp},
    parser, ExParseError, Operator,
};
use num::Float;
use regex::Regex;
use smallvec::{smallvec, SmallVec};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    iter::repeat,
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
impl<'a, T: Copy + Debug> DeepNode<'a, T> where T: Float {
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
    pub fn from_tuple((repr, func): (&'a str, fn(T) -> T)) -> UnaryOpWithReprs<'a, T> {
        let funcs: VecOfUnaryFuncs<T> = smallvec![func];
        UnaryOpWithReprs {
            reprs: vec![repr],
            op: UnaryOp::from_vec(funcs),
        }
    }
    pub fn append_front(&mut self, other: &mut UnaryOpWithReprs<'a, T>) {
        self.op.append_front(&mut other.op);
        self.reprs = other
            .reprs
            .iter()
            .chain(self.reprs.iter())
            .map(|f| *f)
            .collect::<Vec<_>>();
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

impl<'a, T: Copy + Debug> DeepEx<'a, T> {
    /// Evaluates all operators with numbers as operands.
    pub fn compile(&mut self) {
        // change from exression to number if an expression contains only a number
        for node in &mut self.nodes {
            if let DeepNode::Expr(ref e) = node {
                if e.nodes.len() == 1 {
                    match e.nodes[0] {
                        DeepNode::Num(n) => {
                            *node = DeepNode::Num(n);
                        }
                        _ => (),
                    }
                }
            };
        }
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
                        *num_idx_after = *num_idx_after - 1;
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
            match self.nodes[0] {
                DeepNode::Num(n) => {
                    self.nodes[0] = DeepNode::Num(self.unary_op.op.apply(n));
                    self.unary_op.op.clear();
                    self.unary_op.reprs.clear();
                }
                _ => (),
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

            let mut expr = DeepEx {
                nodes: nodes,
                bin_ops: bin_ops,
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
            DeepNode::Var((idx, _)) => format!("{{x{}}}", idx),
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
                res.push_str("(");
                res
            });
        let closings =
            repeat(")")
                .take(self.unary_op.op.len())
                .fold(String::new(), |mut res, closing| {
                    res.push_str(closing);
                    res
                });
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
            unary_op: unary_op,
        }
    }

    pub fn from_str(text: &'a str) -> Result<DeepEx<'a, T>, ExParseError>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Float + FromStr,
    {
        let ops = operators::make_default_operators::<T>();
        Ok(DeepEx::from_ops(&text, &ops)?)
    }

    pub fn from_ops(text: &'a str, ops: &[Operator<'a, T>]) -> Result<DeepEx<'a, T>, ExParseError>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Copy + FromStr + Debug,
    {
        let parsed_tokens = parser::tokenize_and_analyze(text, &ops, parser::is_numeric_text)?;
        let mut deepex = deep_details::parsed_tokens_to_deepex(&parsed_tokens)?;
        deepex.set_overloaded_ops(find_overloaded_ops(ops));
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
        let is_numeric = |text: &'a str| parser::is_numeric_regex(&re_number, &text);
        let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
        let mut deepex = deep_details::parsed_tokens_to_deepex(&parsed_tokens)?;
        deepex.set_overloaded_ops(deep_details::find_overloaded_ops(ops));
        Ok(deepex)
    }

    pub fn set_overloaded_ops(&mut self, ops: Option<OverloadedOps<'a, T>>) {
        self.overloaded_ops = ops;
    }

    fn reset_vars(&mut self, new_var_names: &[&str]) {
        for n in &mut self.nodes {
            match n {
                DeepNode::Expr(e) => e.reset_vars(new_var_names),
                DeepNode::Var((i, n)) => {
                    for (new_idx, new_name) in new_var_names.iter().enumerate() {
                        if n == new_name {
                            *i = new_idx;
                        }
                    }
                }
                _ => (),
            }
        }
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

    pub fn overloaded_ops(&self) -> &Option<OverloadedOps<'a, T>> {
        &self.overloaded_ops
    }
    pub fn var_names(&self) -> &SmallVec<[& 'a str; N_VARS_ON_STACK]> {
        &self.var_names
    }

    pub fn unpack_and_clone_overloaded_ops(&self) -> Result<OverloadedOps<'a, T>, ExParseError> {
        self.overloaded_ops.clone().ok_or(ExParseError {
            msg: "cannot unpack overloaded ops when there are none".to_string(),
        })
    }

    fn is_num(&self, num: T) -> bool where T: Float {
        self.nodes.len() == 1 && match &self.nodes[0] {
            DeepNode::Num(n) => *n == num,
            DeepNode::Expr(e) => e.is_num(num),
            _ => false,
        }
    }

    pub fn is_one(&self) -> bool where T: Float {
        self.is_num(T::from(1.0).unwrap())
    }

    pub fn is_zero(&self) -> bool where T: Float {
        self.is_num(T::from(0.0).unwrap())
    }

    pub fn var_names_union(self, other: Self) -> (Self, Self) {
        let mut all_var_names = self.var_names.clone();
        for name in other.var_names.clone() {
            if !all_var_names.contains(&name) {
                all_var_names.push(name);
            }
        }
        let mut self_vars_updated = self;
        let mut other_vars_updated = other;
        self_vars_updated.reset_vars(&all_var_names);
        other_vars_updated.reset_vars(&all_var_names);
        self_vars_updated.var_names = all_var_names.clone();
        other_vars_updated.var_names = all_var_names;
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
            ops: ops,
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
use super::flat::flatten;
#[cfg(test)]
use crate::{operators::make_default_operators, util::assert_float_eq_f64};


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
        for vn in first.var_names() {
            assert!(var_names.contains(vn));
        }
        for vn in second.var_names() {
            assert!(var_names.contains(vn));
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
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();
    assert_eq!(deep_ex.nodes.len(), 1);
    match deep_ex.nodes[0] {
        DeepNode::Num(n) => assert_eq!(deep_ex.unary_op.op.apply(n), n),
        _ => {
            assert!(false);
        }
    }
}
