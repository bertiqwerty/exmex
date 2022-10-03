use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    str::FromStr,
};

use num::Float;
use smallvec::SmallVec;

use crate::{
    data_type::DataType,
    definitions::{
        N_BINOPS_OF_DEEPEX_ON_STACK, N_NODES_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK,
        N_VARS_ON_STACK,
    },
    expression::flat::ExprIdxVec,
    format_exerr,
    operators::UnaryOp,
    BinOp, ExError, ExResult, Express, FloatOpsFactory, MakeOperators, MatchLiteral, NumberMatcher,
    Operator,
};

#[cfg(feature = "partial")]
use crate::Differentiate;

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;
mod detail {
    use std::{fmt::Debug, iter, str::FromStr};

    use num::Float;
    use smallvec::SmallVec;

    use crate::{
        data_type::DataType,
        definitions::N_BINOPS_OF_DEEPEX_ON_STACK,
        operators::{UnaryOp, VecOfUnaryFuncs},
        parser::{self, Paren, ParsedToken},
        DeepEx, ExError, ExResult, MakeOperators, MatchLiteral,
    };

    use super::{BinOpVec, BinOpsWithReprs, DeepNode, UnaryOpWithReprs};

    pub fn operate_bin<'a, T, OF, LM>(
        deepex1: DeepEx<'a, T, OF, LM>,
        deepex2: DeepEx<'a, T, OF, LM>,
        bin_op: BinOpsWithReprs<'a, T>,
    ) -> DeepEx<'a, T, OF, LM>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        let (self_vars_updated, other_vars_updated) = deepex1.var_names_union(deepex2);
        let mut resex = DeepEx::new(
            vec![
                DeepNode::Expr(Box::new(self_vars_updated)),
                DeepNode::Expr(Box::new(other_vars_updated)),
            ],
            bin_op,
            super::UnaryOpWithReprs::new(),
        )
        .unwrap();
        resex.compile();
        resex
    }

    pub fn is_num<T, OF, LM>(deepex: &DeepEx<T, OF, LM>, num: T) -> bool
    where
        T: DataType + Float,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        deepex.nodes().len() == 1
            && match &deepex.nodes()[0] {
                DeepNode::Num(n) => deepex.unary_op().op.apply(*n) == num,
                DeepNode::Expr(e) => is_num(e, num),
                _ => false,
            }
    }

    pub fn parse<'a, T, F, OF, LM>(text: &'a str, is_numeric: F) -> ExResult<DeepEx<'a, T, OF, LM>>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
        F: Fn(&str) -> Option<&str>,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
    {
        let ops = OF::make();
        let parsed_tokens = parser::tokenize_and_analyze(text, &ops, is_numeric)?;
        parser::check_parsed_token_preconditions(&parsed_tokens)?;
        let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
        let (deepex, _): (DeepEx<'a, T, OF, LM>, usize) =
            make_expression(&parsed_tokens[0..], &parsed_vars, UnaryOpWithReprs::new())?;
        Ok(deepex)
    }
    pub fn unparse_raw<T, OF, LM>(
        nodes: &[DeepNode<T, OF, LM>],
        bin_ops: &BinOpsWithReprs<T>,
        unary_op: &UnaryOpWithReprs<T>,
    ) -> String
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        let mut node_strings = nodes.iter().map(|n| match n {
            DeepNode::Num(n) => format!("{:?}", n),
            DeepNode::Var((_, var_name)) => format!("{{{}}}", var_name),
            DeepNode::Expr(e) => {
                if e.unary_op().op.len() == 0 {
                    format!("({})", unparse_raw(e.nodes(), e.bin_ops(), e.unary_op()))
                } else {
                    unparse_raw(e.nodes(), e.bin_ops(), e.unary_op())
                }
            }
        });
        let mut bin_op_strings = bin_ops.reprs.iter();
        // a valid expression has at least one node
        let first_node_str = node_strings.next().unwrap();
        let node_with_bin_ops_string = node_strings.fold(first_node_str, |mut res, node_str| {
            let bin_op_str = bin_op_strings.next().unwrap();
            res.push_str(bin_op_str);
            res.push_str(node_str.as_str());
            res
        });
        let unary_op_string = unary_op
            .reprs
            .iter()
            .fold(String::new(), |mut res, uop_str| {
                res.push_str(uop_str);
                res.push('(');
                res
            });
        let closings =
            iter::repeat(")")
                .take(unary_op.op.len())
                .fold(String::new(), |mut res, closing| {
                    res.push_str(closing);
                    res
                });
        if unary_op.op.len() == 0 {
            node_with_bin_ops_string
        } else {
            format!(
                "{}{}{}",
                unary_op_string, node_with_bin_ops_string, closings
            )
        }
    }
    /// Handles the case that a token is a unary operator and returns a tuple.
    /// The first element is a node that is either an expression with a unary operator or a
    /// number where the unary operator has been applied to. the second element is the number
    /// of tokens that are covered by the unary operator and its argument. Note that a unary
    /// operator can be a composition of multiple functions.
    pub fn process_unary<'a, T, OF, LM>(
        token_idx: usize,
        unary_op: fn(T) -> T,
        repr: &'a str,
        parsed_tokens: &[ParsedToken<'a, T>],
        parsed_vars: &[&'a str],
    ) -> ExResult<(DeepNode<'a, T, OF, LM>, usize)>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        // gather subsequent unary operators from the beginning
        let iter_of_uops = iter::once(Ok((repr, unary_op))).chain(
            (token_idx + 1..parsed_tokens.len())
                .map(|j| match &parsed_tokens[j] {
                    ParsedToken::Op(op) => {
                        if op.has_unary() {
                            Some(op)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .take_while(|op| op.is_some())
                .map(|op| {
                    let op = op.unwrap();
                    Ok((op.repr(), op.unary()?))
                }),
        );
        let vec_of_uops = iter_of_uops
            .clone()
            .map(|op| Ok(op?.1))
            .collect::<ExResult<VecOfUnaryFuncs<_>>>()?;
        let vec_of_uop_reprs = iter_of_uops
            .clone()
            .map(|op| Ok(op?.0))
            .collect::<ExResult<SmallVec<_>>>()?;
        let n_uops = vec_of_uops.len();
        let uop = UnaryOp::from_vec(vec_of_uops);
        match &parsed_tokens[token_idx + n_uops] {
            ParsedToken::Paren(_) => {
                let (expr, i_forward) = make_expression::<T, OF, LM>(
                    &parsed_tokens[token_idx + n_uops + 1..],
                    parsed_vars,
                    UnaryOpWithReprs {
                        reprs: vec_of_uop_reprs,
                        op: uop,
                    },
                )?;
                Ok((DeepNode::Expr(Box::new(expr)), i_forward + n_uops + 1))
            }
            ParsedToken::Var(name) => {
                let expr = DeepEx::new(
                    vec![DeepNode::Var((
                        parser::find_var_index(name, parsed_vars),
                        name.to_string(),
                    ))],
                    BinOpsWithReprs::new(),
                    UnaryOpWithReprs {
                        reprs: vec_of_uop_reprs,
                        op: uop,
                    },
                )?;
                Ok((DeepNode::Expr(Box::new(expr)), n_uops + 1))
            }
            ParsedToken::Num(n) => Ok((DeepNode::Num(uop.apply(n.clone())), n_uops + 1)),
            _ => Err(ExError::new("Invalid parsed token configuration")),
        }
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
    pub fn make_expression<'a, T, OF, LM>(
        parsed_tokens: &[ParsedToken<'a, T>],
        parsed_vars: &[&'a str],
        unary_ops: UnaryOpWithReprs<'a, T>,
    ) -> ExResult<(DeepEx<'a, T, OF, LM>, usize)>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        let mut bin_ops = BinOpVec::new();
        let mut reprs_bin_ops: SmallVec<[&str; N_BINOPS_OF_DEEPEX_ON_STACK]> = SmallVec::new();
        let mut nodes = Vec::<DeepNode<T, OF, LM>>::new();
        nodes.reserve(parsed_tokens.len() / 2);
        // The main loop checks one token after the next whereby sub-expressions are
        // handled recursively. Thereby, the token-position-index idx_tkn is increased
        // according to the length of the sub-expression.
        let mut idx_tkn: usize = 0;
        while idx_tkn < parsed_tokens.len() {
            match &parsed_tokens[idx_tkn] {
                ParsedToken::Op(op) => {
                    if idx_tkn > 0 && parser::is_operator_binary(op, &parsed_tokens[idx_tkn - 1])? {
                        bin_ops.push(op.bin()?);
                        reprs_bin_ops.push(op.repr());
                        idx_tkn += 1;
                    } else {
                        let (node, idx_forward) = process_unary(
                            idx_tkn,
                            op.unary()?,
                            op.repr(),
                            parsed_tokens,
                            parsed_vars,
                        )?;
                        nodes.push(node);
                        idx_tkn += idx_forward;
                    }
                }
                ParsedToken::Num(n) => {
                    nodes.push(DeepNode::Num(n.clone()));
                    idx_tkn += 1;
                }
                ParsedToken::Var(name) => {
                    nodes.push(DeepNode::Var((
                        parser::find_var_index(name, parsed_vars),
                        name.to_string(),
                    )));
                    idx_tkn += 1;
                }
                ParsedToken::Paren(p) => match p {
                    Paren::Open => {
                        idx_tkn += 1;
                        let (expr, i_forward) = make_expression::<T, OF, LM>(
                            &parsed_tokens[idx_tkn..],
                            parsed_vars,
                            UnaryOpWithReprs::new(),
                        )?;
                        nodes.push(DeepNode::Expr(Box::new(expr)));
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
    /// Correction for cases where nodes are unnecessarily wrapped in expression-nodes.
    pub fn lift_nodes<T, OF, LM>(deepex: &mut DeepEx<T, OF, LM>)
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
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
                                *node = DeepNode::Var(std::mem::take(v));
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
}

fn find_op<'a, T: Clone + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> Option<Operator<'a, T>> {
    ops.iter().cloned().find(|op| op.repr() == repr)
}

pub fn find_bin_op<'a, T: Clone + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> ExResult<BinOpsWithReprs<'a, T>> {
    let op = find_op(repr, ops).ok_or_else(|| format_exerr!("did not find operator {}", repr))?;
    Ok(BinOpsWithReprs {
        reprs: smallvec::smallvec![op.repr()],
        ops: smallvec::smallvec![op.bin()?],
    })
}

pub fn find_unary_op<'a, T: Clone + Debug>(
    repr: &'a str,
    ops: &[Operator<'a, T>],
) -> ExResult<UnaryOpWithReprs<'a, T>> {
    let op = find_op(repr, ops).ok_or_else(|| format_exerr!("did not find operator {}", repr))?;
    Ok(UnaryOpWithReprs {
        reprs: smallvec::smallvec![op.repr()],
        op: UnaryOp::from_vec(smallvec::smallvec![op.unary()?]),
    })
}

pub fn prioritized_indices<T, OF, LM>(
    bin_ops: &[BinOp<T>],
    nodes: &[DeepNode<T, OF, LM>],
) -> ExprIdxVec
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    let prio_increase = |bin_op_idx: usize| match (&nodes[bin_op_idx], &nodes[bin_op_idx + 1]) {
        (DeepNode::Num(_), DeepNode::Num(_)) if bin_ops[bin_op_idx].is_commutative => {
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
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprs<'a, T: Clone> {
    pub reprs: SmallVec<[&'a str; N_BINOPS_OF_DEEPEX_ON_STACK]>,
    pub ops: BinOpVec<T>,
}
impl<'a, T: Clone> BinOpsWithReprs<'a, T> {
    pub fn new() -> Self {
        BinOpsWithReprs {
            reprs: smallvec::smallvec![],
            ops: BinOpVec::new(),
        }
    }
}
impl<'a, T: Clone> Default for BinOpsWithReprs<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOpWithReprs<'a, T> {
    pub reprs: SmallVec<[&'a str; N_UNARYOPS_OF_DEEPEX_ON_STACK]>,
    pub op: UnaryOp<T>,
}
impl<'a, T> UnaryOpWithReprs<'a, T>
where
    T: Clone,
{
    pub fn new() -> UnaryOpWithReprs<'a, T> {
        UnaryOpWithReprs {
            reprs: smallvec::smallvec![],
            op: UnaryOp::new(),
        }
    }

    pub fn append_after(&mut self, other: UnaryOpWithReprs<'a, T>) {
        self.op.append_after(other.op);
        self.reprs = other
            .reprs
            .iter()
            .chain(self.reprs.iter())
            .copied()
            .collect();
    }

    pub fn remove_latest(&mut self) {
        self.op.remove_latest();
        self.reprs.remove(0);
    }

    pub fn clear(&mut self) {
        self.op.clear();
        self.reprs.clear();
    }
}
impl<'a, T: Clone> Default for UnaryOpWithReprs<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}
/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DeepNode<'a, T, OF = FloatOpsFactory<T>, LM = NumberMatcher>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    /// Boxing this due to <https://rust-lang.github.io/rust-clippy/master/index.html#large_enum_variant>
    Expr(Box<DeepEx<'a, T, OF, LM>>),
    Num(T),
    /// The contained integer points to the index of the variable.
    Var((usize, String)),
}
impl<'a, T, OF, LM> DeepNode<'a, T, OF, LM>
where
    T: DataType + Float,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
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
impl<'a, T, OF, LM> Debug for DeepNode<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DeepNode::Expr(e) => write!(f, "{:#?}", e),
            DeepNode::Num(n) => write!(f, "{:?}", n),
            DeepNode::Var((_, var_name)) => write!(f, "{}", var_name),
        }
    }
}

/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T, OF = FloatOpsFactory<T>, LM = NumberMatcher>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T, OF, LM>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: BinOpsWithReprs<'a, T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: UnaryOpWithReprs<'a, T>,
    var_names: SmallVec<[String; N_VARS_ON_STACK]>,

    text: String,
    dummy_ops_factory: PhantomData<OF>,
    dummy_literal_matcher_factory: PhantomData<LM>,
}

impl<'a, T, OF, LM> DeepEx<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    pub fn make_ops(&self) -> Vec<Operator<'a, T>> {
        OF::make()
    }

    /// Compiles expression, needed for partial differentation.
    pub fn compile(&mut self) {
        detail::lift_nodes(self);

        let prio_indices = prioritized_indices(&self.bin_ops.ops, &self.nodes);
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
        self.text = detail::unparse_raw(self.nodes(), self.bin_ops(), self.unary_op());
    }

    pub fn new(
        nodes: Vec<DeepNode<'a, T, OF, LM>>,
        bin_ops: BinOpsWithReprs<'a, T>,
        unary_op: UnaryOpWithReprs<'a, T>,
    ) -> ExResult<DeepEx<'a, T, OF, LM>> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(format_exerr!(
                "mismatch between number of nodes {:?} and binary operators {:?} ({} vs {})",
                nodes,
                bin_ops.ops,
                nodes.len(),
                bin_ops.ops.len()
            ))
        } else {
            let mut found_vars = SmallVec::<[String; N_VARS_ON_STACK]>::new();
            for node in &nodes {
                match node {
                    DeepNode::Num(_) => (),
                    DeepNode::Var((_, name)) => {
                        // see https://github.com/rust-lang/rust/issues/42671
                        // and https://doc.rust-lang.org/std/vec/struct.Vec.html#method.contains
                        // on why Iterator::any instead of Vec::contains
                        if !found_vars.iter().any(|v| v == name) {
                            found_vars.push(name.to_string());
                        }
                    }
                    DeepNode::Expr(e) => {
                        for name in &e.var_names {
                            if !found_vars.iter().any(|v| v == name) {
                                found_vars.push(name.to_string());
                            }
                        }
                    }
                }
            }
            found_vars.sort_unstable();
            let var_names = found_vars.iter().map(|s| s.to_string()).collect();
            let text = "not yet compiled".to_string();
            let mut expr = DeepEx {
                nodes,
                bin_ops,
                unary_op,
                var_names,
                text,
                dummy_ops_factory: PhantomData,
                dummy_literal_matcher_factory: PhantomData,
            };
            expr.compile();
            expr.text = detail::unparse_raw(expr.nodes(), expr.bin_ops(), expr.unary_op());
            Ok(expr)
        }
    }

    pub fn from_node(node: DeepNode<'a, T, OF, LM>) -> DeepEx<'a, T, OF, LM> {
        DeepEx::new(vec![node], BinOpsWithReprs::new(), UnaryOpWithReprs::new()).unwrap()
    }

    pub fn one() -> DeepEx<'a, T, OF, LM>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::one())
    }

    pub fn zero() -> DeepEx<'a, T, OF, LM>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::zero())
    }

    pub fn from_num(x: T) -> DeepEx<'a, T, OF, LM>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::num(x))
    }

    pub fn without_latest_unary_op(mut self) -> Self {
        self.unary_op.remove_latest();
        self
    }
    pub fn with_new_latest_unary_op(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.remove_latest();
        self.unary_op.append_after(unary_op);
        self
    }

    pub fn with_only_unary_op(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
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

    pub fn nodes(&self) -> &Vec<DeepNode<'a, T, OF, LM>> {
        &self.nodes
    }

    pub fn is_num(&self, num: T) -> bool
    where
        T: Float,
    {
        detail::is_num(self, num)
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

    pub fn reset_vars(&mut self, new_var_names: SmallVec<[String; N_VARS_ON_STACK]>) {
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

    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    pub fn var_names_union(self, other: Self) -> (Self, Self) {
        let mut all_var_names = self.var_names.iter().cloned().collect::<SmallVec<_>>();
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

    pub fn var_names_like_other(mut self, other: &Self) -> Self {
        self.var_names = other.var_names.clone();
        self
    }

    /// Applies a binary operator to self and other
    pub fn operate_bin_repr(self, other: Self, bin_op_repr: &'a str) -> ExResult<Self> {
        let ops = OF::make();
        let bin_op = find_bin_op(bin_op_repr, &ops)?;
        Ok(detail::operate_bin(self, other, bin_op))
    }

    /// Applies a binary operator to self and other
    pub fn operate_bin(self, other: Self, bin_op: BinOpsWithReprs<'a, T>) -> Self {
        detail::operate_bin(self, other, bin_op)
    }

    /// Applies a unary operator to self
    pub fn operate_unary_repr(mut self, unary_op_repr: &'a str) -> ExResult<Self> {
        let ops = OF::make();
        let unary_op = find_unary_op(unary_op_repr, &ops)?;
        self.unary_op.append_after(unary_op);
        self.compile();
        Ok(self)
    }
    /// Applies a unary operator to self
    pub fn operate_unary(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.append_after(unary_op);
        self.compile();
        self
    }
}

impl<'a, T, OF, LM> Express<'a, T> for DeepEx<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    type LiteralMatcher = LM;
    type OperatorFactory = OF;
    fn eval_relaxed(&self, vars: &[T]) -> ExResult<T> {
        if self.var_names().len() > vars.len() {
            return Err(format_exerr!(
                "expression contains {} vars which is different to the length {} of the passed slice",
                self.var_names.len(),
                vars.len()
            ));
        }
        let mut numbers = self
            .nodes()
            .iter()
            .map(|node| -> ExResult<T> {
                match node {
                    DeepNode::Num(n) => Ok(n.clone()),
                    DeepNode::Var((idx, _)) => Ok(vars[*idx].clone()),
                    DeepNode::Expr(e) => e.eval_relaxed(vars),
                }
            })
            .collect::<ExResult<SmallVec<[T; N_NODES_ON_STACK]>>>()?;
        let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec::smallvec![false; self.nodes().len()];
        let prio_indices = prioritized_indices(&self.bin_ops().ops, self.nodes());
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
            numbers[num_idx - shift_left] = (self.bin_ops().ops[bin_op_idx].apply)(num_1, num_2);
            ignore[num_idx + shift_right] = true;
        }
        Ok(self.unary_op().op.apply(numbers[0].clone()))
    }
    fn eval(&self, vars: &[T]) -> ExResult<T> {
        if self.var_names().len() != vars.len() {
            return Err(format_exerr!(
                "expression contains {} vars which is different to the length {} of the passed slice",
                self.var_names.len(),
                vars.len()
            ));
        }
        self.eval_relaxed(vars)
    }
    fn from_deepex(deepex: DeepEx<'a, T, OF, LM>) -> ExResult<DeepEx<'a, T, OF, LM>>
    where
        Self: Sized,
        T: DataType,
        <T as FromStr>::Err: Debug,
    {
        Ok(deepex)
    }
    fn to_deepex(&self) -> ExResult<DeepEx<'a, T, OF, LM>>
    where
        Self: Sized,
        T: DataType,
        <T as FromStr>::Err: Debug,
    {
        Ok(self.clone())
    }
    fn unparse(&self) -> &str {
        self.text.as_str()
    }
    fn var_names(&self) -> &[String] {
        &self.var_names
    }
    fn parse(text: &'a str) -> ExResult<Self>
    where
        Self: Sized,
    {
        detail::parse(text, LM::is_literal)
    }
}

impl<'a, T, OF, LM> Display for DeepEx<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.unparse())
    }
}

#[cfg(feature = "partial")]
impl<'a, T, OF, LM> Differentiate<'a, T> for DeepEx<'a, T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T> + Debug,
    LM: MatchLiteral + Debug,
    <T as FromStr>::Err: Debug,
{
}

#[cfg(test)]
use crate::{operators::VecOfUnaryFuncs, util::assert_float_eq_f64, FlatEx};
#[test]
fn test_reset_vars() {
    let deepex = DeepEx::<f64>::parse("2*z+x+y * .5").unwrap();
    let ref_vars = ["x", "y", "z"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex.var_names()[i], *rv);
    }
    let deepex2 = DeepEx::parse("a*c*b").unwrap();
    let ref_vars = ["a", "b", "c"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex2.var_names()[i], *rv);
    }
    let (deepex_, deepex2_) = deepex.clone().var_names_union(deepex2.clone());
    let all_vars = ["a", "b", "c", "x", "y", "z"];
    for (i, av) in all_vars.iter().enumerate() {
        assert_eq!(deepex_.var_names()[i], *av);
        assert_eq!(deepex2_.var_names()[i], *av);
    }
    assert_eq!(deepex.unparse(), deepex_.unparse());
    assert_eq!(deepex2.unparse(), deepex2_.unparse());
}

#[test]
fn test_var_name_union() -> ExResult<()> {
    fn test(str_1: &str, str_2: &str, var_names: &[&str]) -> ExResult<()> {
        let first = DeepEx::<f64>::parse(str_1)?;
        let second = DeepEx::<f64>::parse(str_2)?;
        let (first, second) = first.var_names_union(second);

        assert_eq!(first.var_names().len(), var_names.len());
        assert_eq!(second.var_names().len(), var_names.len());
        for vn in first.var_names() {
            assert!(var_names.contains(&vn.as_str()));
        }
        for vn in second.var_names() {
            assert!(var_names.contains(&vn.as_str()));
        }
        Ok(())
    }

    test("x", "y", &["x", "y"])?;
    test("x+y*z", "z+y", &["x", "y", "z"])?;
    Ok(())
}

#[test]
fn test_var_names() {
    let deepex = DeepEx::<f64>::parse("x+y+{x}+z*(-y)").unwrap();
    assert_eq!(deepex.var_names()[0], "x");
    assert_eq!(deepex.var_names()[1], "y");
    assert_eq!(deepex.var_names()[2], "z");
}

#[test]
fn test_deep_compile() {
    let ops = FloatOpsFactory::<f64>::make();
    let nodes = vec![
        DeepNode::<f64>::Num(4.5),
        DeepNode::Num(0.5),
        DeepNode::Num(1.4),
    ];
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
    assert_eq!(deepex.nodes().len(), 1);
    match deepex.nodes()[0] {
        DeepNode::Num(n) => assert_float_eq_f64(deepex.unary_op().op.apply(n), n),
        _ => {
            unreachable!();
        }
    }
}

#[test]
fn test_deep_lift_node() {
    let deepex =
        DeepEx::<f64>::parse("(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))").unwrap();
    println!("{}", deepex);
    assert_eq!(
        format!("{}", deepex),
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))"
    );

    let deepex = DeepEx::<f64>::parse("(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "{a}+{x}^2.0*{x}^2.0");
}

#[test]
fn test_deep_compile_2() {
    let expr = DeepEx::<f64>::parse("1.0 * 3 * 2 * x / 2 / 3").unwrap();
    assert_float_eq_f64(expr.eval(&[2.0]).unwrap(), 2.0);
    let expr = DeepEx::<f64>::parse(
        "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+2+3+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
    )
    .unwrap();
    assert_eq!(
        "{x}*0.25+{x}*8.0+5.0+7.0*sin({y})-{z}/sin(1.5/(1.0-{x}*4.0))",
        expr.unparse()
    );
    let expr = DeepEx::<f64>::parse("x + 1 - 2").unwrap();
    assert_float_eq_f64(expr.eval(&[0.0]).unwrap(), -1.0);
    let expr = DeepEx::<f64>::parse("x - 1 + 2").unwrap();
    assert_float_eq_f64(expr.eval(&[0.0]).unwrap(), 1.0);
    let expr = DeepEx::<f64>::parse("x * 2 / 3").unwrap();
    assert_float_eq_f64(expr.eval(&[2.0]).unwrap(), 4.0 / 3.0);
    let expr = DeepEx::<f64>::parse("x / 2 / 3").unwrap();
    assert_float_eq_f64(expr.eval(&[2.0]).unwrap(), 1.0 / 3.0);
}

#[test]
fn test_operate_unary() -> ExResult<()> {
    let lstr = "x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)";
    let deepex = DeepEx::<f64>::parse(lstr)?;
    let mut funcs = VecOfUnaryFuncs::new();
    funcs.push(|x: f64| x * 1.23456);
    let deepex = deepex.operate_unary(UnaryOpWithReprs {
        reprs: smallvec::smallvec!["eagle"],
        op: UnaryOp::from_vec(funcs),
    });
    assert_float_eq_f64(deepex.eval(&[1.0, 1.75, 2.25])?, -0.23148000000000002 * 8.0);
    Ok(())
}

#[test]
fn test_unparse() -> ExResult<()> {
    fn test(text: &str, text_ref: &str) -> ExResult<()> {
        let flatex = FlatEx::<f64>::parse(text)?;
        assert_eq!(flatex.unparse(), text);
        let deepex = DeepEx::<f64>::parse(text)?;
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
