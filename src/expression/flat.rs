use crate::data_type::DataType;
use crate::definitions::{N_NODES_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK};
use crate::expression::flat_details::{self, FlatNodeKind, FlatNodeVec, FlatOpVec};
use crate::expression::{
    deep::{DeepBuf, DeepEx, ExprIdxVec},
    partial_derivatives, Express,
};
use crate::operators::UnaryOp;
use crate::parser::{Paren, ParsedToken};
use crate::{parser, ExError, ExResult, FloatOpsFactory, MakeOperators, Operator};
use num::Float;
use regex::Regex;
use smallvec::SmallVec;
use std::fmt::{self, Debug, Display, Formatter};
use std::marker::PhantomData;
use std::str::FromStr;

type UnaryOpIdxDepthStack = SmallVec<[(usize, i64); N_UNARYOPS_OF_DEEPEX_ON_STACK]>;

const DEPTH_PRIO_STEP: i64 = 1000;

/// This is called in case a closing paren occurs. If available, the index of the unary operator of the
/// relevant depth operators will be returned and the open operator will be removed.
///   
fn pop_unary_stack(unary_stack: &mut UnaryOpIdxDepthStack, depth: i64) -> Option<usize> {
    let last_idx_depth = unary_stack.last().copied();
    match last_idx_depth {
        Some((idx, d)) if d == depth => {
            unary_stack.pop();
            Some(idx)
        }
        _ => None,
    }
}

fn is_binary<'a, T>(op: &Operator<'a, T>, idx: usize, parsed_tokens: &[ParsedToken<'a, T>]) -> bool
where
    T: DataType,
{
    idx > 0 && parser::is_operator_binary(op, &parsed_tokens[idx - 1])
}

fn unpack_unary<'a, T>(
    idx: usize,
    parsed_tokens: &[ParsedToken<'a, T>],
) -> ExResult<Option<fn(T) -> T>>
where
    T: DataType,
{
    match &parsed_tokens[idx] {
        ParsedToken::Op(op) => {
            if !is_binary(op, idx, parsed_tokens) {
                Ok(Some(op.unary()?))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

pub fn make_expression<'a, T, OF>(
    text: &'a str,
    parsed_tokens: &[ParsedToken<'a, T>],
    parsed_vars: &[&'a str],
) -> ExResult<FlatEx<'a, T, OF>>
where
    T: Clone + FromStr + Debug,
    OF: MakeOperators<T>,
{
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    let mut idx_tkn: usize = 0;
    let mut depth = 0;
    let mut unary_stack: UnaryOpIdxDepthStack = SmallVec::new();

    let iter_subsequent_unaries = |end_idx: usize| {
        let unpack = |idx| unpack_unary(idx, parsed_tokens);
        let dist_from_end = (0..end_idx + 1)
            .rev()
            .map(unpack)
            .take_while(|f| match f {
                Ok(f) => f.is_some(),
                _ => false,
            })
            .count();
        let start_idx = end_idx + 1 - dist_from_end;

        // check if we did terminate due to an error
        if start_idx > 0 {
            unpack(start_idx - 1)?;
        }

        Ok((start_idx..end_idx + 1).map(unpack).flatten().flatten())
    };

    let create_node = |idx_node, kind| {
        if idx_node > 0 {
            let idx_op = idx_node - 1;
            if let ParsedToken::Op(op) = &parsed_tokens[idx_op] {
                if !is_binary(op, idx_op, parsed_tokens) {
                    return Ok(FlatNode {
                        kind,
                        unary_op: UnaryOp::from_iter(iter_subsequent_unaries(idx_op)?),
                    });
                }
            }
        }
        Ok(FlatNode::from_kind(kind))
    };
    while idx_tkn < parsed_tokens.len() {
        match &parsed_tokens[idx_tkn] {
            ParsedToken::Op(op) => {
                if is_binary(op, idx_tkn, parsed_tokens) {
                    let mut bin_op = op.bin()?;
                    bin_op.prio += depth * DEPTH_PRIO_STEP;
                    flat_ops.push(FlatOp::<T> {
                        unary_op: UnaryOp::new(),
                        bin_op,
                    });
                } else if let ParsedToken::Paren(p) = &parsed_tokens[idx_tkn + 1] {
                    let err_msg = "a unary operator cannot on the left of a closing paren";
                    match p {
                        Paren::Close => return Err(ExError::new(err_msg)),
                        Paren::Open => unary_stack.push((idx_tkn, depth)),
                    };
                }
                idx_tkn += 1;
            }
            ParsedToken::Num(n) => {
                let kind = FlatNodeKind::Num(n.clone());
                let flat_node = create_node(idx_tkn, kind)?;
                flat_nodes.push(flat_node);
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                let idx = parser::find_var_index(name, parsed_vars);
                let kind = FlatNodeKind::Var(idx);
                let flat_node = create_node(idx_tkn, kind)?;
                flat_nodes.push(flat_node);
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => {
                match p {
                    Paren::Open => {
                        idx_tkn += 1;
                        depth += 1;
                    }
                    Paren::Close => {
                        let lowest_prio_flat_op = flat_ops
                            .iter_mut()
                            .rev()
                            .take_while(|op| op.bin_op.prio >= depth * DEPTH_PRIO_STEP)
                            .min_by(|fo1, fo2| fo1.bin_op.prio.cmp(&fo2.bin_op.prio));
                        match lowest_prio_flat_op {
                            None => {
                                // no binary operators of current depth, attach to last node
                                let last_node = flat_nodes.iter_mut().last().ok_or_else(|| {
                                    ExError::new("there must be a node between parens")
                                })?;
                                let mut closed = pop_unary_stack(&mut unary_stack, depth - 1);
                                match &mut closed {
                                    None => (),
                                    Some(uop_idx) => last_node
                                        .unary_op
                                        .append_after_iter(iter_subsequent_unaries(*uop_idx)?),
                                }
                            }
                            Some(lowpfo) => {
                                let mut closed = pop_unary_stack(&mut unary_stack, depth - 1);
                                match &mut closed {
                                    None => (),
                                    Some(uop_idx) => lowpfo
                                        .unary_op
                                        .append_after_iter(iter_subsequent_unaries(*uop_idx)?),
                                }
                            }
                        }
                        idx_tkn += 1;
                        depth -= 1;
                    }
                }
            }
        }
    }
    let indices = flat_details::prioritized_indices_flat(&flat_ops, &flat_nodes);
    Ok(FlatEx {
        nodes: flat_nodes,
        ops: flat_ops,
        prio_indices: indices,
        n_unique_vars: parsed_vars.len(),
        deepex: None,
        text: Some(text),
        dummy: PhantomData,
    })
}

fn parse<'a, T, OF, F>(
    text: &'a str,
    ops: &[Operator<'a, T>],
    is_numeric: F,
) -> ExResult<FlatEx<'a, T, OF>>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    F: Fn(&'a str) -> Option<&'a str>,
    OF: MakeOperators<T>,
{
    let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
    parser::check_parsed_token_preconditions(&parsed_tokens)?;
    let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
    let mut expr = make_expression(text, &parsed_tokens[0..], &parsed_vars)?;
    expr.compile();
    Ok(expr)
}

/// Parses a string directly into a [`FlatEx`](FlatEx) without compilation of the expression.
/// Serialization and partial differentiation is not possible when using fast parsing.
pub fn fast_parse<T>(text: &str) -> ExResult<FlatEx<T>>
where
    T: DataType + num::Float,
    <T as FromStr>::Err: Debug,
{
    let default_ops = FloatOpsFactory::make();
    parse(text, &default_ops, parser::is_numeric_text)
}

/// This is the core data type representing a flattened expression and the result of
/// parsing a string. We use flattened expressions to make efficient evaluation possible.
/// Simplified, a flat expression consists of a [`SmallVec`](https://docs.rs/smallvec/)
/// of nodes and a [`SmallVec`](https://docs.rs/smallvec/) of operators that are applied
/// to the nodes in an order following operator priorities.
///
/// Creation of expressions is possible with the function [`parse`](crate::parse) which is equivalent to
/// [`FlatEx::from_str`](FlatEx::from_str) or with [`FlatEx::from_pattern`](FlatEx::from_pattern).
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::prelude::*;
///
/// // create an expression by parsing a string
/// let expr = FlatEx::<f32>::from_str("sin(1+y)*x")?;
/// assert!((expr.eval(&[1.5, 2.0])? - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The argument `&[1.5, 2.0]` in the call of [`eval`](FlatEx::eval) specifies the
/// variable values in the alphabetical order of the variable names.
/// In this example, we want to evaluate the expression for the varibale values `x=2.0` and `y=1.5`.
///
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatEx<'a, T, OF = FloatOpsFactory<T>>
where
    T: Clone + Debug,
    OF: MakeOperators<T>,
{
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
    deepex: Option<DeepEx<'a, T>>,
    text: Option<&'a str>,
    dummy: PhantomData<OF>,
}

impl<'a, T, OF> FlatEx<'a, T, OF>
where
    T: Clone + Debug,
    OF: MakeOperators<T>,
{
    fn flatten(deepex: DeepEx<'a, T>) -> Self {
        let (nodes, ops) = flat_details::flatten_vecs(&deepex, 0);
        let indices = flat_details::prioritized_indices_flat(&ops, &nodes);
        let n_unique_vars = deepex.n_vars();
        Self {
            nodes,
            ops,
            prio_indices: indices,
            n_unique_vars,
            deepex: Some(deepex),
            text: None,
            dummy: PhantomData,
        }
    }

    fn compile(&mut self) {
        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();

        let mut already_declined: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec::smallvec![false; self.nodes.len()];

        for node in &mut self.nodes {
            match &node.kind {
                FlatNodeKind::Num(num) => {
                    *node =
                        FlatNode::from_kind(FlatNodeKind::Num(node.unary_op.apply(num.clone())));
                }
                _ => (),
            }
        }
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (FlatNodeKind::Num(num_1), FlatNodeKind::Num(num_2)) =
                (node_1.kind.clone(), node_2.kind.clone())
            {
                if !(already_declined[num_idx] || already_declined[num_idx + 1]) {
                    let op_result =
                        self.ops[bin_op_idx]
                            .unary_op
                            .apply((self.ops[bin_op_idx].bin_op.apply)(num_1, num_2));
                    self.nodes[num_idx] = FlatNode::from_kind(FlatNodeKind::Num(op_result));
                    self.nodes.remove(num_idx + 1);
                    already_declined.remove(num_idx + 1);
                    // reduce indices after removed position
                    for num_idx_after in num_inds.iter_mut() {
                        if *num_idx_after > num_idx {
                            *num_idx_after -= 1;
                        }
                    }
                    used_prio_indices.push(bin_op_idx);
                } else {
                    already_declined[num_idx] = true;
                    already_declined[num_idx + 1] = true;
                }
            } 
            else {
                already_declined[num_idx] = true;
                already_declined[num_idx + 1] = true;
            }
        }

        self.ops = self
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|(_, op)| op.clone())
            .collect();

        self.prio_indices = flat_details::prioritized_indices_flat(&self.ops, &self.nodes);
    }
}

impl<'a, T, OF> Express<'a, T> for FlatEx<'a, T, OF>
where
    T: DataType,
    OF: MakeOperators<T>,
{
    fn from_str(text: &'a str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        let ops = OF::make();
        parse(text, &ops, parser::is_numeric_text)
    }

    fn from_regex(text: &'a str, number_regex: &Regex) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        let ops = OF::make();
        let is_numeric = |text: &'a str| parser::is_numeric_regex(number_regex, text);
        parse(text, &ops, is_numeric)
    }

    fn from_pattern(text: &'a str, number_regex_pattern: &str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        let re_number = match Regex::new(number_regex_pattern) {
            Ok(regex) => regex,
            Err(_) => {
                return Err(ExError {
                    msg: "Cannot compile the passed number regex.".to_string(),
                })
            }
        };
        Self::from_regex(text, &re_number)
    }

    fn eval(&self, vars: &[T]) -> ExResult<T> {
        flat_details::eval_flatex(
            vars,
            &self.nodes,
            &self.ops,
            &self.prio_indices,
            self.n_unique_vars,
        )
    }
    fn partial(mut self, var_idx: usize) -> ExResult<Self>
    where
        T: DataType + Float,
        <T as FromStr>::Err: Debug,
    {
        check_partial_index(var_idx, self.n_vars(), self.unparse()?.as_str())?;
        let ops = FloatOpsFactory::make();

        if self.deepex.is_none() {
            self.deepex = match self.text {
                Some(t) => Some(DeepEx::from_ops(t, &OF::make())?),
                None => {
                    return Err(ExError::new(
                        "Need either text or deep expression. Did you call `reduce_memory`?",
                    ));
                }
            }
        }

        let d_i = partial_derivatives::partial_deepex(
            var_idx,
            self.deepex
                .expect("This is bug. deepex cannot be None here."),
            &ops,
        )?;
        Ok(Self::flatten(d_i))
    }
    fn unparse(&self) -> ExResult<String> {
        match self.text {
            Some(t) => Ok(t.to_string()),
            None => match &self.deepex {
                Some(deepex) => Ok(deepex.unparse_raw()),
                None => Err(ExError {
                    msg: "unparse impossible, since deep expression optimized away".to_string(),
                }),
            },
        }
    }
    fn reduce_memory(&mut self) {
        self.deepex = None;
    }

    fn n_vars(&self) -> usize {
        self.n_unique_vars
    }
}

/// The expression is displayed as a string created by [`unparse`](FlatEx::unparse).
impl<'a, T, OF> Display for FlatEx<'a, T, OF>
where
    T: DataType,
    OF: MakeOperators<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unparsed = self.unparse();
        match unparsed {
            Err(e) => write!(f, "{}", e.msg),
            Ok(s) => write!(f, "{}", s),
        }
    }
}

/// This is another representation of a flattened expression besides [`FlatEx`](FlatEx).
/// The difference is that [`OwnedFlatEx`](OwnedFlatEx) can be used without
/// a lifetime parameter. All the data that [`FlatEx`](FlatEx) borrowed is kept in a
/// buffer by [`OwnedFlatEx`](OwnedFlatEx). The drawback is that parsing takes longer, since
/// additional allocations are necessary. Evaluation time should be about the same for
/// [`FlatEx`](FlatEx) and [`OwnedFlatEx`](OwnedFlatEx).
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{Express, OwnedFlatEx};
/// let to_be_parsed = "log(z) + 2* (-z^(x-2) + sin(4*y))";
/// let expr_owned = OwnedFlatEx::<f64>::from_str(to_be_parsed)?;
/// assert!((expr_owned.eval(&[4.0, 3.7, 2.5])? - 14.992794866624788 as f64).abs() < 1e-12);
/// #
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct OwnedFlatEx<T, OF = FloatOpsFactory<T>>
where
    T: Clone + Debug,
    OF: MakeOperators<T>,
{
    deepex_buf: Option<DeepBuf<T>>,
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
    text: Option<String>,
    dummy: PhantomData<OF>,
}
impl<T, OF> OwnedFlatEx<T, OF>
where
    T: Clone + Debug,
    OF: MakeOperators<T>,
{
    /// Creates an `OwnedFlatEx` instance from an instance of `FlatEx`.
    pub fn from_flatex(flatex: FlatEx<T, OF>) -> Self {
        Self {
            deepex_buf: flatex.deepex.map(|d| DeepBuf::from_deepex(&d)),
            nodes: flatex.nodes,
            ops: flatex.ops,
            prio_indices: flatex.prio_indices,
            n_unique_vars: flatex.n_unique_vars,
            text: flatex.text.map(|s| s.to_string()),
            dummy: PhantomData,
        }
    }
}
impl<'a, T, OF> Express<'a, T> for OwnedFlatEx<T, OF>
where
    T: DataType,
    OF: MakeOperators<T>,
{
    fn from_str(text: &'a str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: Clone + FromStr,
    {
        Ok(Self::from_flatex(FlatEx::from_str(text)?))
    }

    fn from_regex(text: &'a str, number_regex: &Regex) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        Ok(Self::from_flatex(FlatEx::from_regex(text, number_regex)?))
    }

    fn from_pattern(text: &'a str, number_regex_pattern: &str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        Ok(Self::from_flatex(FlatEx::from_pattern(
            text,
            number_regex_pattern,
        )?))
    }

    fn eval(&self, vars: &[T]) -> ExResult<T> {
        flat_details::eval_flatex(
            vars,
            &self.nodes,
            &self.ops,
            &self.prio_indices,
            self.n_unique_vars,
        )
    }

    fn partial(mut self, var_idx: usize) -> ExResult<Self>
    where
        T: Float,
        <T as FromStr>::Err: Debug,
    {
        check_partial_index(var_idx, self.n_vars(), self.unparse()?.as_str())?;

        let ops = FloatOpsFactory::make();

        if self.deepex_buf.is_none() {
            self.deepex_buf = match self.text {
                Some(t) => {
                    let deepex = DeepEx::from_ops(t.as_str(), &OF::make())?;
                    Some(DeepBuf::from_deepex(&deepex))
                }
                None => {
                    return Err(ExError::new(
                        "Need either text or deep expression. Did you call `reduce_memory`?",
                    ));
                }
            };
        }

        let deep_buf = self
            .deepex_buf
            .expect("This is bug. deepex buffer cannot be None here.");
        let deepex = deep_buf.to_deepex(&ops)?;
        let d_i = partial_derivatives::partial_deepex(var_idx, deepex, &ops)?;
        Ok(Self::from_flatex(FlatEx::<T, OF>::flatten(d_i)))
    }
    fn unparse(&self) -> ExResult<String> {
        match &self.text {
            Some(t) => Ok(t.clone()),
            None => match &self.deepex_buf {
                Some(deepex) => Ok(deepex.unparsed.clone()),
                None => Err(ExError {
                    msg: "unparse impossible, since deep expression optimized away".to_string(),
                }),
            },
        }
    }

    fn reduce_memory(&mut self) {
        self.deepex_buf = None;
    }

    fn n_vars(&self) -> usize {
        self.n_unique_vars
    }
}
/// The expression is displayed as a string created by [`unparse`](OwnedFlatEx::unparse).
impl<T, OF> Display for OwnedFlatEx<T, OF>
where
    T: DataType,
    OF: MakeOperators<T>,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unparsed = self.unparse();
        match unparsed {
            Err(e) => write!(f, "{}", e.msg),
            Ok(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(test)]
use crate::{
    expression::deep::{self, UnaryOpWithReprs},
    operators::VecOfUnaryFuncs,
    util::assert_float_eq_f64,
};
#[cfg(test)]
use smallvec::smallvec;

use super::flat_details::{check_partial_index, FlatNode, FlatOp};

#[test]
fn test_fast_parse() -> ExResult<()> {
    fn test(sut: &str, vars: &[f64], reference: f64) -> ExResult<()> {
        println!("  ===  testing {}", sut);
        let flatex = fast_parse::<f64>(sut)?;
        println!("{:#?}", flatex);
        assert_float_eq_f64(flatex.eval(vars)?, reference);
        Ok(())
    }
    test("sin(1)", &[], 1.0.sin())?;
    test("2*3^2", &[], 2.0 * 3.0.powi(2))?;
    test("sin(-(sin(2)))*2", &[], (-(2f64.sin())).sin() * 2.0)?;
    test("sin(-(0.7))", &[], (-0.7).sin())?;
    test("sin(-0.7)", &[], (-0.7).sin())?;
    test("sin(-x)", &[0.7], (-0.7).sin())?;
    test("1.3+(-0.7)", &[], 0.6)?;
    test("2-1/2", &[], 2.0 - 1.0 / 2.0)?;
    test("log(log2(2))*tan(2)+exp(1.5)", &[], 4.4816890703380645)?;
    test("sin(0)", &[], 0f64.sin())?;
    test("1-(1-2)", &[], 2.0)?;
    test("1-(1-x)", &[2.0], 2.0)?;
    test("1*sin(2-0.1) + x", &[1.0], 1.0 + 1.9f64.sin())?;
    test("sin(6)", &[], -0.27941549819892586)?;
    test("sin(x+2)", &[5.0], 0.6569865987187891)?;
    test("sin((x+1))", &[5.0], -0.27941549819892586)?;
    test("sin(y^(x+1))", &[5.0, 2.0], 0.9200260381967907)?;
    test("sin(((a*y^(x+1))))", &[0.5, 5.0, 2.0], 0.5514266812416906)?;
    test(
        "sin(((cos((a*y^(x+1))))))",
        &[0.5, 5.0, 2.0],
        0.7407750251209115,
    )?;
    test("sin(cos(x+1))", &[5.0], 0.819289219220601)?;
    test(
        "5*{χ} +  4*log2(log(1.5+γ))*({χ}*-(tan(cos(sin(652.2-{γ}))))) + 3*{χ}",
        &[1.2, 1.0],
        8.040556934857268,
    )?;
    test(
        "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)",
        &[1.5, 0.2532],
        -3.1164569260604176,
    )?;
    test("sin(x)+sin(y)+sin(z)", &[1.0, 2.0, 3.0], 1.8918884196934453)?;
    test("x*0.2*5.0/4.0+x*2.0*4.0*1.0*1.0*1.0*1.0*1.0*1.0*1.0+7.0*sin(y)-z/sin(3.0/2.0/(1.0-x*4.0*1.0*1.0*1.0*1.0))",
    &[1.0, 2.0, 3.0], 20.872570916580237)?;
    test("sin(-(1.0))", &[], -0.8414709848078965)?;
    test("x*0.02*(3-(2*y))", &[1.0, 2.0], -0.02)?;
    test("x*((x*1)-0.98)*(0.5*-y)", &[1.0, 2.0], -0.02)?;
    test("x*0.02*sin(3*(2*y))", &[1.0, 2.0], 0.02 * (12.0).sin())?;
    test(
        "x*0.02*sin(-(3.0*(2.0*sin(x-1.0/(sin(y*5.0)+(5.0-1.0/z))))))",
        &[1.0, 2.0, 3.0],
        0.01661860154948708,
    )?;
    Ok(())
}

#[test]
fn test_operate_unary() {
    let lstr = "x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)";
    let deepex = deep::from_str(lstr).unwrap();
    let mut funcs = VecOfUnaryFuncs::new();
    funcs.push(|x: f64| x * 1.23456);
    let deepex = deepex.operate_unary(UnaryOpWithReprs {
        reprs: smallvec!["eagle"],
        op: UnaryOp::from_vec(funcs),
    });
    let flatex = FlatEx::<f64>::flatten(deepex);
    assert_float_eq_f64(
        flatex.eval(&[1.0, 1.75, 2.25]).unwrap(),
        -0.23148000000000002 * 8.0,
    );
}

#[test]
fn test_flat_clear() -> ExResult<()> {
    let flatex = FlatEx::<f64>::from_str("x*(2*(2*(2*4*8)))")?;
    assert_float_eq_f64(flatex.eval(&[1.0])?, 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    let mut deri = flatex.partial(0)?;
    assert!(deri.deepex.is_some());
    deri.reduce_memory();
    assert!(deri.deepex.is_none());

    let flatex = OwnedFlatEx::<f64>::from_str("x*(2*(2*(2*4*8)))")?;
    assert_float_eq_f64(flatex.eval(&[1.0])?, 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    let mut deri = flatex.partial(0)?;
    assert!(deri.deepex_buf.is_some());
    deri.reduce_memory();
    assert!(deri.deepex_buf.is_none());
    Ok(())
}

#[test]
fn test_flat_compile() -> ExResult<()> {
    fn test(text: &str, vars: &[f64], ref_val: f64, ref_len: usize) -> ExResult<()> {
        println!("testing {}...", text);
        let flatex = FlatEx::<f64>::from_str(text)?;
        assert_float_eq_f64(flatex.eval(vars)?, ref_val);
        assert_eq!(flatex.nodes.len(), ref_len);
        let flatex = OwnedFlatEx::<f64>::from_flatex(flatex);
        assert_float_eq_f64(flatex.eval(vars)?, ref_val);
        assert_eq!(flatex.nodes.len(), ref_len);
        let flatex = OwnedFlatEx::<f64>::from_str(text)?;
        assert_float_eq_f64(flatex.eval(vars)?, ref_val);
        assert_eq!(flatex.nodes.len(), ref_len);
        println!("...ok.");
        Ok(())
    }

    test("1*sin(2-0.1)", &[], 1.9f64.sin(), 1)?;
    test("x*(2*(2*(2*4*8)))", &[1.0], 32.0 * 8.0, 2)?;
    test("1*sin(2-0.1) + x", &[1.0], 1.0 + 1.9f64.sin(), 2)?;
    test("1.0 * 3 * 2 * x / 2 / 3", &[2.0], 2.0, 4)?;
    test(
        "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+2+3+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
        &[2.21, 2.0, 3.0],
        45.37365538326699,
        13,
    )?;
    test("x / 2 / 3", &[1.0], 1.0/6.0, 3)?;
    test("x * 2 / 3", &[1.0], 2.0/3.0, 2)?;
    let flatex = FlatEx::<f64>::from_str("1*sin(2-0.1) + x")?;
    match flatex.nodes[0].kind {
        FlatNodeKind::Num(n) => assert_float_eq_f64(n, 1.9f64.sin()),
        _ => unreachable!(),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => unreachable!(),
    }

    let flatex = OwnedFlatEx::<f64>::from_str("y + 1 - cos(1/(1*sin(2-0.1))-2) + 2 + x")?;
    assert_eq!(flatex.nodes.len(), 3);
    match flatex.nodes[0].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 1),
        _ => unreachable!(),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Num(_) => (),
        _ => unreachable!(),
    }
    match flatex.nodes[2].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => unreachable!(),
    }
    Ok(())
}

#[test]
fn test_display() {
    let mut flatex = FlatEx::<f64>::from_str("sin(var)/5").unwrap();
    println!("{}", flatex);
    assert_eq!(format!("{}", flatex), "sin(var)/5");
    flatex.reduce_memory();
    assert_eq!(format!("{}", flatex), "sin(var)/5");

    let flatex = FlatEx::<f64>::from_str("sin(var)/5").unwrap();
    let mut owned_flatex = OwnedFlatEx::from_flatex(flatex);
    assert_eq!(format!("{}", owned_flatex), "sin(var)/5");
    owned_flatex.reduce_memory();
    assert_eq!(format!("{}", owned_flatex), "sin(var)/5");
}

#[test]
fn test_unparse() -> ExResult<()> {
    fn test(text: &str, text_ref: &str) -> ExResult<()> {
        let mut flatex = FlatEx::<f64>::from_str(text)?;
        assert_eq!(flatex.unparse()?, text);
        flatex.reduce_memory();
        assert!(flatex.unparse().is_ok());

        let mut flatex = OwnedFlatEx::<f64>::from_str(text)?;
        assert_eq!(flatex.unparse()?, text);
        flatex.reduce_memory();
        assert!(flatex.unparse().is_ok());

        let deepex = DeepEx::<f64>::from_ops(text, &FloatOpsFactory::make())?;
        assert_eq!(deepex.unparse_raw(), text_ref);
        Ok(())
    }
    let text = "5+x";
    let text_ref = "5.0+{x}";
    test(text, text_ref)?;
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
