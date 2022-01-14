use crate::{
    definitions::{N_BINOPS_OF_DEEPEX_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK},
    expression::flat::ExprIdxVec,
    format_exerr,
    operators::{BinOp, UnaryOp},
    partial::{BinOpVec, DeepEx, DeepNode},
    ExError, ExResult,
};
use std::{fmt::Debug, iter};

use num::Float;
use smallvec::SmallVec;

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

    pub fn append_after(&mut self, other: &UnaryOpWithReprs<'a, T>) {
        self.op.append_after(&other.op);
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

pub fn prioritized_indices<T: Clone + Debug>(
    bin_ops: &[BinOp<T>],
    nodes: &[DeepNode<T>],
) -> ExprIdxVec {
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

pub fn unparse_raw<T: Clone + Debug>(deepex: &DeepEx<T>) -> String {
    let mut node_strings = deepex.nodes().iter().map(|n| match n {
        DeepNode::Num(n) => format!("{:?}", n),
        DeepNode::Var((_, var_name)) => format!("{{{}}}", var_name),
        DeepNode::Expr(e) => {
            if e.unary_op().op.len() == 0 {
                format!("({})", e.unparse())
            } else {
                e.unparse()
            }
        }
    });
    let mut bin_op_strings = deepex.bin_ops().reprs.iter();
    // a valid expression has at least one node
    let first_node_str = node_strings.next().unwrap();
    let node_with_bin_ops_string = node_strings.fold(first_node_str, |mut res, node_str| {
        let bin_op_str = bin_op_strings.next().unwrap();
        res.push_str(bin_op_str);
        res.push_str(node_str.as_str());
        res
    });
    let unary_op_string = deepex
        .unary_op()
        .reprs
        .iter()
        .fold(String::new(), |mut res, uop_str| {
            res.push_str(uop_str);
            res.push('(');
            res
        });
    let closings = iter::repeat(")").take(deepex.unary_op().op.len()).fold(
        String::new(),
        |mut res, closing| {
            res.push_str(closing);
            res
        },
    );
    if deepex.unary_op().op.len() == 0 {
        node_with_bin_ops_string
    } else {
        format!(
            "{}{}{}",
            unary_op_string, node_with_bin_ops_string, closings
        )
    }
}

pub fn operate_bin<'a, T: Clone + Debug>(
    deepex1: DeepEx<'a, T>,
    deepex2: DeepEx<'a, T>,
    bin_op: BinOpsWithReprs<'a, T>,
) -> DeepEx<'a, T> {
    let (self_vars_updated, other_vars_updated) = deepex1.var_names_union(deepex2);
    let mut resex = DeepEx::new(
        vec![
            DeepNode::Expr(Box::new(self_vars_updated)),
            DeepNode::Expr(Box::new(other_vars_updated)),
        ],
        bin_op,
        UnaryOpWithReprs::new(),
    )
    .unwrap();
    resex.compile();
    resex
}

pub fn is_num<T: Clone + Debug>(deepex: &DeepEx<T>, num: T) -> bool
where
    T: Float,
{
    deepex.nodes().len() == 1
        && match &deepex.nodes()[0] {
            DeepNode::Num(n) => deepex.unary_op().op.apply(*n) == num,
            DeepNode::Expr(e) => is_num(e, num),
            _ => false,
        }
}

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

#[cfg(test)]
use {
    crate::{
        operators::VecOfUnaryFuncs,
        parser::{self, Paren, ParsedToken},
    },
    std::str::FromStr,
};

/// Handles the case that a token is a unary operator and returns a tuple.
/// The first element is a node that is either an expression with a unary operator or a
/// number where the unary operator has been applied to. the second element is the number
/// of tokens that are covered by the unary operator and its argument. Note that a unary
/// operator can be a composition of multiple functions.
#[cfg(test)]
fn process_unary<'a, T: Clone + FromStr + Debug>(
    token_idx: usize,
    unary_op: fn(T) -> T,
    repr: &'a str,
    parsed_tokens: &[ParsedToken<'a, T>],
    parsed_vars: &[&'a str],
) -> ExResult<(DeepNode<'a, T>, usize)> {
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
            let (expr, i_forward) = make_expression::<T>(
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
                    name,
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
#[cfg(test)]
pub fn make_expression<'a, T>(
    parsed_tokens: &[ParsedToken<'a, T>],
    parsed_vars: &[&'a str],
    unary_ops: UnaryOpWithReprs<'a, T>,
) -> ExResult<(DeepEx<'a, T>, usize)>
where
    T: Clone + FromStr + Debug,
{
    let mut bin_ops = BinOpVec::new();
    let mut reprs_bin_ops: SmallVec<[&'a str; N_BINOPS_OF_DEEPEX_ON_STACK]> = SmallVec::new();
    let mut nodes = Vec::<DeepNode<T>>::new();
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
                    let (node, idx_forward) =
                        process_unary(idx_tkn, op.unary()?, op.repr(), parsed_tokens, parsed_vars)?;
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
                    name,
                )));
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => match p {
                Paren::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) = make_expression::<T>(
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
