use crate::definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK};
use crate::{
    operators,
    operators::{BinOp, UnaryOp, VecOfUnaryFuncs},
    parser,
    parser::{Paren, ParsedToken},
    ExParseError, Operator,
};
use num::Float;
use regex::Regex;
use smallvec::{smallvec, SmallVec};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    iter::{once, repeat},
    ops::{Add, Div, Mul, Sub},
    str::FromStr,
};

pub type ExprIdxVec = SmallVec<[usize; N_NODES_ON_STACK]>;

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;

const ADD_REPR: &str = "+";
const SUB_REPR: &str = "-";
const MUL_REPR: &str = "*";
const DIV_REPR: &str = "/";

fn parsed_tokens_to_deepex<'a, T: Copy + FromStr + Debug>(
    parsed_tokens: &[ParsedToken<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
    let mut found_vars = SmallVec::<[&str; N_VARS_ON_STACK]>::new();
    let parsed_vars = parsed_tokens
        .iter()
        .filter_map(|pt| match pt {
            ParsedToken::Var(name) => {
                if !found_vars.contains(name) {
                    found_vars.push(*name);
                    Some(*name)
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect::<SmallVec<[_; N_NODES_ON_STACK]>>();

    let (expr, _) = make_expression(
        &parsed_tokens[0..],
        &parsed_vars,
        UnaryOpWithReprs {
            reprs: vec![],
            op: UnaryOp::new(),
        },
    )?;
    Ok(expr)
}

/// Returns an expression that is created recursively and can be evaluated
///
/// # Arguments
///
/// * `parsed_tokens` - parsed tokens created with [`tokenize_and_analyze`](parse::tokenize_and_analyze)
/// * `parsed_vars` - elements of `parsed_tokens` that are variables
/// * `unary_ops` - unary operators of the expression to be build
///
/// # Errors
///
/// See [`parse_with_number_pattern`](parse_with_number_pattern)
///
fn make_expression<'a, T>(
    parsed_tokens: &[ParsedToken<'a, T>],
    parsed_vars: &[&'a str],
    unary_ops: UnaryOpWithReprs<'a, T>,
) -> Result<(DeepEx<'a, T>, usize), ExParseError>
where
    T: Copy + FromStr + Debug,
{
    fn unpack_binop<S>(bo: Option<BinOp<S>>) -> BinOp<S>
    where
        S: Copy + FromStr + Debug,
    {
        match bo {
            Some(bo) => bo,
            None => panic!("This is probably a bug. Expected binary operator but there was none."),
        }
    }

    let find_var_index = |name: &str| {
        let idx = parsed_vars.iter().enumerate().find(|(_, n)| **n == name);
        match idx {
            Some((i, _)) => i,
            None => {
                panic!("This is probably a bug. I don't know variable {}", name)
            }
        }
    };
    // this closure handles the case that a token is a unary operator and accesses the
    // variable 'tokens' from the outer scope
    let process_unary = |i: usize, uo, repr| {
        // gather subsequent unary operators from the beginning
        let iter_of_uops = once((repr, uo)).chain(
            (i + 1..parsed_tokens.len())
                .map(|j| match parsed_tokens[j] {
                    ParsedToken::Op(op) => (op.repr, op.unary_op),
                    _ => ("", None),
                })
                .take_while(|(_, uo_)| uo_.is_some())
                .map(|(repr_, uo_)| (repr_, uo_.unwrap())),
        );
        let vec_of_uops = iter_of_uops
            .clone()
            .map(|(_, uo_)| uo_)
            .collect::<VecOfUnaryFuncs<_>>();
        let vec_of_uop_reprs = iter_of_uops
            .clone()
            .map(|(repr_, _)| repr_)
            .collect::<Vec<_>>();
        let n_uops = vec_of_uops.len();
        let uop = UnaryOp::from_vec(vec_of_uops);
        match &parsed_tokens[i + n_uops] {
            ParsedToken::Paren(p) => match p {
                Paren::Close => Err(ExParseError {
                    msg: "closing parenthesis after an operator".to_string(),
                }),
                Paren::Open => {
                    let (expr, i_forward) = make_expression::<T>(
                        &parsed_tokens[i + n_uops + 1..],
                        &parsed_vars,
                        UnaryOpWithReprs {
                            reprs: vec_of_uop_reprs,
                            op: uop,
                        },
                    )?;
                    Ok((DeepNode::Expr(expr), i_forward + n_uops + 1))
                }
            },
            ParsedToken::Var(name) => {
                let expr = DeepEx::new(
                    vec![DeepNode::Var((find_var_index(name), name))],
                    BinOpsWithReprs {
                        reprs: Vec::new(),
                        ops: BinOpVec::new(),
                    },
                    UnaryOpWithReprs {
                        reprs: vec_of_uop_reprs,
                        op: uop,
                    },
                )?;
                Ok((DeepNode::Expr(expr), n_uops + 1))
            }
            ParsedToken::Num(n) => Ok((DeepNode::Num(uop.apply(*n)), n_uops + 1)),
            ParsedToken::Op(_) => Err(ExParseError {
                msg: "a unary operator cannot be followed by a binary operator".to_string(),
            }),
        }
    };

    let mut bin_ops = BinOpVec::new();
    let mut reprs_bin_ops: Vec<&str> = Vec::new();
    let mut nodes = Vec::<DeepNode<T>>::new();

    // The main loop checks one token after the next whereby sub-expressions are
    // handled recursively. Thereby, the token-position-index idx_tkn is increased
    // according to the length of the sub-expression.
    let mut idx_tkn: usize = 0;
    while idx_tkn < parsed_tokens.len() {
        match &parsed_tokens[idx_tkn] {
            ParsedToken::Op(op) => match op.unary_op {
                None => {
                    bin_ops.push(unpack_binop(op.bin_op));
                    reprs_bin_ops.push(op.repr);
                    idx_tkn += 1;
                }
                Some(uo) => {
                    // might the operator be unary?
                    if idx_tkn == 0 {
                        // if the first element is an operator it must be unary
                        let (node, idx_forward) = process_unary(idx_tkn, uo, op.repr)?;
                        nodes.push(node);
                        idx_tkn += idx_forward;
                    } else {
                        // decide type of operator based on predecessor
                        match &parsed_tokens[idx_tkn - 1] {
                            ParsedToken::Num(_) | ParsedToken::Var(_) => {
                                // number or variable as predecessor means binary operator
                                bin_ops.push(unpack_binop(op.bin_op));
                                reprs_bin_ops.push(op.repr);
                                idx_tkn += 1;
                            }
                            ParsedToken::Paren(p) => match p {
                                Paren::Open => {
                                    let msg = "This is probably a bug. An opening paren cannot be the predecessor of a binary operator.";
                                    panic!("{}", msg);
                                }
                                Paren::Close => {
                                    bin_ops.push(unpack_binop(op.bin_op));
                                    reprs_bin_ops.push(op.repr);
                                    idx_tkn += 1;
                                }
                            },
                            ParsedToken::Op(_) => {
                                let (node, idx_forward) = process_unary(idx_tkn, uo, op.repr)?;
                                nodes.push(node);
                                idx_tkn += idx_forward;
                            }
                        }
                    }
                }
            },
            ParsedToken::Num(n) => {
                nodes.push(DeepNode::Num(*n));
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                nodes.push(DeepNode::Var((find_var_index(name), name)));
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => match p {
                Paren::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) = make_expression::<T>(
                        &parsed_tokens[idx_tkn..],
                        &parsed_vars,
                        UnaryOpWithReprs {
                            reprs: Vec::new(),
                            op: UnaryOp::new(),
                        },
                    )?;
                    nodes.push(DeepNode::Expr(expr));
                    idx_tkn += i_forward;
                }
                Paren::Close => {
                    idx_tkn += 1;
                    break;
                }
            },
        }
    }
    Ok((
        DeepEx::new(
            nodes,
            BinOpsWithReprs {
                reprs: reprs_bin_ops,
                ops: bin_ops,
            },
            unary_ops,
        )?,
        idx_tkn,
    ))
}

fn prioritized_indices<T: Copy + Debug>(bin_ops: &[BinOp<T>], nodes: &[DeepNode<T>]) -> ExprIdxVec {
    let prio_increase = |bin_op_idx: usize| match (&nodes[bin_op_idx], &nodes[bin_op_idx + 1]) {
        (DeepNode::Num(_), DeepNode::Num(_)) => {
            let prio_inc = 5;
            &bin_ops[bin_op_idx].prio * 10 + prio_inc
        }
        _ => &bin_ops[bin_op_idx].prio * 10,
    };

    let mut indices: ExprIdxVec = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let prio_i1 = prio_increase(*i1);
        let prio_i2 = prio_increase(*i2);
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

fn find_overloaded_ops<'a, T: Copy>(
    all_ops: &[Operator<T>],
) -> Result<OverloadedOps<'a, T>, ExParseError> {
    let find_op = |repr| {
        let found = all_ops.iter().cloned().find(|op| op.repr == repr);
        match found {
            Some(op) => Some(Operator {
                bin_op: op.bin_op,
                unary_op: op.unary_op,
                repr: repr,
            }),
            None => None,
        }
    };

    let make_err = |repr| ExParseError {
        msg: format!("did not find overloaded operator {}", repr),
    };

    Ok(OverloadedOps {
        add: find_op(ADD_REPR).ok_or(make_err(ADD_REPR))?,
        sub: find_op(SUB_REPR).ok_or(make_err(SUB_REPR))?,
        mul: find_op(MUL_REPR).ok_or(make_err(MUL_REPR))?,
        div: find_op(DIV_REPR).ok_or(make_err(DIV_REPR))?,
    })
}



/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum DeepNode<'a, T: Copy + Debug> {
    Expr(DeepEx<'a, T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var((usize, &'a str)),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprs<'a, T: Copy> {
    pub reprs: Vec<&'a str>,
    pub ops: BinOpVec<T>,
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
            .map(|f| *f)
            .collect::<Vec<_>>();
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct OverloadedOps<'a, T: Copy> {
    pub add: Operator<'a, T>,
    pub sub: Operator<'a, T>,
    pub mul: Operator<'a, T>,
    pub div: Operator<'a, T>,
}
impl<'a, T: Copy> OverloadedOps<'a, T> {
    pub fn by_repr(&self, repr: &str) -> Operator<'a, T> {
        match repr {
            ADD_REPR => self.add,
            SUB_REPR => self.sub,
            MUL_REPR => self.mul,
            DIV_REPR => self.div,
            _ => panic!("{} is not a repr of an overloaded operator", repr),
        }
    }
}
/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T: Copy + Debug> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: BinOpsWithReprs<'a, T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: UnaryOpWithReprs<'a, T>,
    prio_indices: ExprIdxVec,
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
        // after changing from expressions to numbers where possible the prios might change
        self.prio_indices = prioritized_indices(&self.bin_ops.ops, &self.nodes);

        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
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

        self.bin_ops.ops = self
            .bin_ops
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|x| *x.1)
            .collect();

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
        self.prio_indices = prioritized_indices(&self.bin_ops.ops, &self.nodes);
    }

    pub fn new(
        nodes: Vec<DeepNode<'a, T>>,
        bin_ops: BinOpsWithReprs<'a, T>,
        unary_op: UnaryOpWithReprs<'a, T>,
    ) -> Result<DeepEx<'a, T>, ExParseError> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(ExParseError {
                msg: "mismatch between number of nodes and binary operators".to_string(),
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

            let indices = prioritized_indices(&bin_ops.ops, &nodes);
            let mut expr = DeepEx {
                nodes: nodes,
                bin_ops: bin_ops,
                unary_op,
                prio_indices: indices,
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

    pub fn set_overloaded_ops(&mut self, overloaded_ops: OverloadedOps<'a, T>) {
        self.overloaded_ops = Some(overloaded_ops);
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
        let mut deepex = parsed_tokens_to_deepex(&parsed_tokens)?;
        let overloaded_ops = find_overloaded_ops(ops);
        match overloaded_ops {
            Err(_) => (),
            Ok(ops) => deepex.set_overloaded_ops(ops),
        }
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
        let mut deepex = parsed_tokens_to_deepex(&parsed_tokens)?;
        let overloaded_ops = find_overloaded_ops(ops);
        match overloaded_ops {
            Err(_) => (),
            Ok(ops) => deepex.set_overloaded_ops(ops),
        }
        Ok(deepex)
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

    pub fn bin_ops(&self) -> &BinOpVec<T> {
        &self.bin_ops.ops
    }

    pub fn unary_op(&self) -> &UnaryOp<T> {
        &self.unary_op.op
    }

    pub fn nodes(&self) -> &Vec<DeepNode<'a, T>> {
        &self.nodes
    }

    /// Applies a binary operator to self and other
    pub fn operate_bin(self, other: Self, bin_op: BinOpsWithReprs<'a, T>) -> Self {
        let mut all_var_names = self.var_names.clone();
        for name in other.var_names.clone() {
            if !all_var_names.contains(&name) {
                all_var_names.push(name);
            }
        }
        let overloaded_ops = self.overloaded_ops.clone();
        let mut self_vars_updated = self;
        let mut other_vars_updated = other;
        self_vars_updated.reset_vars(&all_var_names);
        other_vars_updated.reset_vars(&all_var_names);
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
use crate::make_default_operators;

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