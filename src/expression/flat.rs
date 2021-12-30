use crate::data_type::DataType;
use crate::definitions::N_UNARYOPS_OF_DEEPEX_ON_STACK;
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

type UnaryOpIdxDepthPairs = SmallVec<[(usize, i64); N_UNARYOPS_OF_DEEPEX_ON_STACK]>;

pub fn make_expression<'a, T, OF>(
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
    let depth_step: i64 = 1000;
    let mut depth = 0;
    let mut open_unary_funcs: UnaryOpIdxDepthPairs = SmallVec::new();
    let is_binary = |op, idx| idx > 0 && parser::is_operator_binary(op, &parsed_tokens[idx - 1]);

    let iter_subsequent_unaries = |end_idx| {
        (0..end_idx + 1)
            .rev()
            .map(|idx| match &parsed_tokens[idx] {
                ParsedToken::Op(op) => {
                    if !is_binary(op, idx) {
                        Some(op.unary().unwrap())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .take_while(|f| f.is_some())
            .flatten()
    };

    let close_open_unary = |ouf_depth_pairs: &mut UnaryOpIdxDepthPairs, depth: i64| {
        let last_open_idx = ouf_depth_pairs
            .iter()
            .filter(|(_, d)| *d == depth)
            .map(|(idx, _)| *idx)
            .last();
        ouf_depth_pairs.retain(|(_, d)| *d != depth);
        last_open_idx
    };

    let create_node = |idx_node, kind| {
        if idx_node > 0 {
            let idx_op = idx_node - 1;
            if let ParsedToken::Op(op) = &parsed_tokens[idx_op] {
                if !is_binary(op, idx_op) {
                    return FlatNode {
                        kind,
                        unary_op: UnaryOp::from_vec(iter_subsequent_unaries(idx_op).collect()),
                    };
                }
            }
        }
        FlatNode::from_kind(kind)
    };
    while idx_tkn < parsed_tokens.len() {
        match &parsed_tokens[idx_tkn] {
            ParsedToken::Op(op) => {
                if is_binary(op, idx_tkn) {
                    let mut bin_op = op.bin()?;
                    bin_op.prio += depth * depth_step;
                    flat_ops.push(FlatOp::<T> {
                        unary_op: UnaryOp::new(),
                        bin_op,
                    });
                } else if let ParsedToken::Paren(p) = &parsed_tokens[idx_tkn + 1] {
                    let err_msg = "a unary operator cannot on the left of a closing paren";
                    match p {
                        Paren::Close => return Err(ExError::new(err_msg)),
                        Paren::Open => open_unary_funcs.push((idx_tkn, depth)),
                    };
                }
                idx_tkn += 1;
            }
            ParsedToken::Num(n) => {
                let kind = FlatNodeKind::Num(n.clone());
                let flat_node = create_node(idx_tkn, kind);
                flat_nodes.push(flat_node);
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                let idx = parser::find_var_index(name, parsed_vars);
                let kind = FlatNodeKind::Var(idx);
                let flat_node = create_node(idx_tkn, kind);
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
                            .take_while(|op| op.bin_op.prio >= depth * depth_step)
                            .min_by(|fo1, fo2| fo1.bin_op.prio.cmp(&fo2.bin_op.prio));
                        match lowest_prio_flat_op {
                            None => {
                                // no binary operators of current depth, attach to last node
                                let last_node = flat_nodes.iter_mut().last().ok_or_else(|| {
                                    ExError::new("there must be a node between parens")
                                })?;
                                let mut closed = close_open_unary(&mut open_unary_funcs, depth - 1);
                                match &mut closed {
                                    None => (),
                                    Some(uop_idx) => last_node
                                        .unary_op
                                        .append_after_iter(iter_subsequent_unaries(*uop_idx)),
                                }
                            }
                            Some(lowpfo) => {
                                let mut closed = close_open_unary(&mut open_unary_funcs, depth - 1);
                                match &mut closed {
                                    None => (),
                                    Some(uop_idx) => lowpfo
                                        .unary_op
                                        .append_after_iter(iter_subsequent_unaries(*uop_idx)),
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
    OF: MakeOperators<T> + Debug,
{
    let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
    parser::check_parsed_token_preconditions(&parsed_tokens)?;
    let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
    let res = make_expression(&parsed_tokens[0..], &parsed_vars)?;
    Ok(res)
}

pub fn fast_parse<T>(text: &str) -> ExResult<FlatEx<T>>
where
    T: DataType + num::Float,
    <T as FromStr>::Err: Debug,
{
    let default_ops = FloatOpsFactory::make();
    parse::<T, FloatOpsFactory<T>, _>(text, &default_ops, parser::is_numeric_text)
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
            dummy: PhantomData,
        }
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
        let deepex = DeepEx::from_ops(text, &ops)?;
        Ok(Self::flatten(deepex))
    }

    fn from_regex(text: &'a str, number_regex: &Regex) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        let ops = OF::make();
        let deepex = DeepEx::from_regex(text, &ops, number_regex)?;
        Ok(Self::flatten(deepex))
    }

    fn from_pattern(text: &'a str, number_regex_pattern: &str) -> ExResult<Self>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        let ops = OF::make();
        let deepex = DeepEx::from_pattern(text, &ops, number_regex_pattern)?;
        Ok(Self::flatten(deepex))
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
    fn partial(self, var_idx: usize) -> ExResult<Self>
    where
        T: Float,
    {
        check_partial_index(var_idx, self.n_vars(), self.unparse()?.as_str())?;
        let ops = FloatOpsFactory::make();

        let d_i = partial_derivatives::partial_deepex(
            var_idx,
            self.deepex.ok_or(ExError {
                msg: "need deep expression for derivation, not possible after calling `reduce_memory`"
                    .to_string(),
            })?,
            &ops,
        )?;
        Ok(Self::flatten(d_i))
    }
    fn unparse(&self) -> ExResult<String> {
        match &self.deepex {
            Some(deepex) => Ok(deepex.unparse_raw()),
            None => Err(ExError {
                msg: "unparse impossible, since deep expression optimized away".to_string(),
            }),
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

    fn partial(self, var_idx: usize) -> ExResult<Self>
    where
        T: Float,
    {
        check_partial_index(var_idx, self.n_vars(), self.unparse()?.as_str())?;

        let ops = FloatOpsFactory::make();
        let deep_buf = match self.deepex_buf {
            Some(d) => Ok(d),
            None => Err(ExError {
                msg: "need deep expression for derivation, not possible after calling `clear`"
                    .to_string(),
            }),
        }?;
        let deepex = deep_buf.to_deepex(&ops)?;
        let d_i = partial_derivatives::partial_deepex(var_idx, deepex, &ops)?;
        Ok(Self::from_flatex(FlatEx::<T, OF>::flatten(d_i)))
    }

    fn unparse(&self) -> ExResult<String> {
        match &self.deepex_buf {
            Some(deepex) => Ok(deepex.unparsed.clone()),
            None => Err(ExError {
                msg: "unparse impossible, since deep expression optimized away".to_string(),
            }),
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
fn test_fast_parse() {
    fn test(sut: &str, vars: &[f64], reference: f64) {
        println!("  ===  testing {}", sut);
        let flatex = fast_parse::<f64>(sut).unwrap();
        assert_float_eq_f64(flatex.eval(vars).unwrap(), reference);
    }
    test("sin(1)", &[], 1.0.sin());
    test("2*3^2", &[], 2.0 * 3.0.powi(2));
    test("sin(-(sin(2)))*2", &[], (-(2f64.sin())).sin() * 2.0);
    test("sin(-(0.7))", &[], (-0.7).sin());
    test("sin(-0.7)", &[], (-0.7).sin());
    test("sin(-x)", &[0.7], (-0.7).sin());
    test("1.3+(-0.7)", &[], 0.6);
    test("2-1/2", &[], 2.0 - 1.0 / 2.0);
    test("log(log2(2))*tan(2)+exp(1.5)", &[], 4.4816890703380645);
    test("sin(0)", &[], 0f64.sin());
    test("1-(1-2)", &[], 2.0);
    test("1-(1-x)", &[2.0], 2.0);
    test("1*sin(2-0.1) + x", &[1.0], 1.0 + 1.9f64.sin());
    test("sin(6)", &[], -0.27941549819892586);
    test("sin(x+2)", &[5.0], 0.6569865987187891);
    test("sin((x+1))", &[5.0], -0.27941549819892586);
    test("sin(y^(x+1))", &[5.0, 2.0], 0.9200260381967907);
    test("sin(((a*y^(x+1))))", &[0.5, 5.0, 2.0], 0.5514266812416906);
    test(
        "sin(((cos((a*y^(x+1))))))",
        &[0.5, 5.0, 2.0],
        0.7407750251209115,
    );
    test("sin(cos(x+1))", &[5.0], 0.819289219220601);
    test(
        "5*{χ} +  4*log2(log(1.5+γ))*({χ}*-(tan(cos(sin(652.2-{γ}))))) + 3*{χ}",
        &[1.2, 1.0],
        8.040556934857268,
    );
    test(
        "5*sin(x * (4-y^(2-x) * 3 * cos(x-2*(y-1/(y-2*1/cos(sin(x*y))))))*x)",
        &[1.5, 0.2532],
        -3.1164569260604176,
    );
    test("sin(x)+sin(y)+sin(z)", &[1.0, 2.0, 3.0], 1.8918884196934453);
    test("x*0.2*5.0/4.0+x*2.0*4.0*1.0*1.0*1.0*1.0*1.0*1.0*1.0+7.0*sin(y)-z/sin(3.0/2.0/(1.0-x*4.0*1.0*1.0*1.0*1.0))",
    &[1.0, 2.0, 3.0], 20.872570916580237);
    test("sin(-(1.0))", &[], -0.8414709848078965);
    test(
        "x*0.02*sin(-(3.0*(2.0*sin(x-1.0/(sin(y*5.0)+(5.0-1.0/z))))))",
        &[1.0, 2.0, 3.0],
        0.01661860154948708,
    );
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
fn test_flat_clear() {
    let mut flatex = FlatEx::<f64>::from_str("x*(2*(2*(2*4*8)))").unwrap();
    assert!(flatex.deepex.is_some());
    flatex.reduce_memory();
    assert!(flatex.deepex.is_none());
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flatex.nodes.len(), 2);
    let mut flatex = OwnedFlatEx::<f64>::from_str("x*(2*(2*(2*4*8)))").unwrap();
    assert!(flatex.deepex_buf.is_some());
    flatex.reduce_memory();
    assert!(flatex.deepex_buf.is_none());
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flatex.nodes.len(), 2);
}
#[test]
fn test_flat_compile() {
    let flatex = FlatEx::<f64>::from_str("1*sin(2-0.1)").unwrap();
    assert_float_eq_f64(flatex.eval(&[]).unwrap(), 1.9f64.sin());
    assert_eq!(flatex.nodes.len(), 1);

    let flatex = FlatEx::<f64>::from_str("x*(2*(2*(2*4*8)))").unwrap();
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flatex.nodes.len(), 2);

    let flatex = FlatEx::<f64>::from_str("1*sin(2-0.1) + x").unwrap();
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 1.0 + 1.9f64.sin());
    assert_eq!(flatex.nodes.len(), 2);
    match flatex.nodes[0].kind {
        FlatNodeKind::Num(n) => assert_float_eq_f64(n, 1.9f64.sin()),
        _ => unreachable!(),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => unreachable!(),
    }

    let flatex = OwnedFlatEx::<f64>::from_str("y + 1 - cos(1/(1*sin(2-0.1))-2) + 2 + x").unwrap();
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
}

#[test]
fn test_display() {
    let mut flatex = FlatEx::<f64>::from_str("sin(var)/5").unwrap();
    println!("{}", flatex);
    assert_eq!(format!("{}", flatex), "sin({var})/5.0");
    flatex.reduce_memory();
    assert_eq!(
        format!("{}", flatex),
        "unparse impossible, since deep expression optimized away"
    );

    let flatex = FlatEx::<f64>::from_str("sin(var)/5").unwrap();
    let mut owned_flatex = OwnedFlatEx::from_flatex(flatex);
    assert_eq!(format!("{}", owned_flatex), "sin({var})/5.0");
    owned_flatex.reduce_memory();
    assert_eq!(
        format!("{}", owned_flatex),
        "unparse impossible, since deep expression optimized away"
    );
}

#[test]
fn test_unparse() {
    fn test(text: &str, text_ref: &str) {
        let flatex = FlatEx::<f64>::from_str(text).unwrap();
        let deepex = flatex.deepex.unwrap();
        assert_eq!(deepex.unparse_raw(), text_ref);
        let mut flatex_reparsed = FlatEx::<f64>::from_str(text).unwrap();
        assert_eq!(flatex_reparsed.unparse().unwrap(), text_ref);
        flatex_reparsed.reduce_memory();
        assert!(flatex_reparsed.unparse().is_err());
    }
    let text = "5+x";
    let text_ref = "5.0+{x}";
    test(text, text_ref);
    let text = "sin(5+var)^(1/{y})+{var}";
    let text_ref = "sin(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref);
    let text = "-(5+var)^(1/{y})+{var}";
    let text_ref = "-(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref);
    let text = "cos(sin(-(5+var)^(1/{y})))+{var}";
    let text_ref = "cos(sin(-(5.0+{var})^(1.0/{y})))+{var}";
    test(text, text_ref);
    let text = "cos(sin(-5+var^(1/{y})))-{var}";
    let text_ref = "cos(sin(-5.0+{var}^(1.0/{y})))-{var}";
    test(text, text_ref);
    let text = "cos(sin(-z+var*(1/{y})))+{var}";
    let text_ref = "cos(sin(-({z})+{var}*(1.0/{y})))+{var}";
    test(text, text_ref);
}
