use crate::{
    definitions::{N_BINOPS_OF_DEEPEX_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK, N_VARS_ON_STACK},
    expression::deep::{BinOpVec, BinOpsWithReprs, DeepEx, DeepNode, ExprIdxVec, UnaryOpWithReprs},
    operators::{BinOp, Operate, UnaryOp, VecOfUnaryFuncs},
    parser::{Paren, ParsedToken},
    ExError, ExResult,
};
use std::{fmt::Debug, iter, str::FromStr};

use smallvec::SmallVec;

pub fn find_parsed_vars<'a, T: Copy + FromStr + Debug, O: Operate<'a, T>>(
    parsed_tokens: &[ParsedToken<'a, T, O>],
) -> SmallVec<[&'a str; N_VARS_ON_STACK]> {
    let mut found_vars = SmallVec::<[&str; N_VARS_ON_STACK]>::new();
    let mut parsed_vars = parsed_tokens
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
        .collect::<SmallVec<[_; N_VARS_ON_STACK]>>();
    parsed_vars.sort_unstable();
    parsed_vars
}

fn is_operator_binary<'a, T: Copy + FromStr, O: Operate<'a, T>>(
    op: &O,
    parsed_token_on_the_left: &ParsedToken<'a, T, O>,
) -> bool {
    if op.has_bin() && !op.has_unary() {
        true
    } else if op.has_bin() && op.has_unary() {
        match parsed_token_on_the_left {
            ParsedToken::Num(_) | ParsedToken::Var(_) | ParsedToken::Paren(_) => true,
            ParsedToken::Op(_) => false,
        }
    } else {
        false
    }
}

fn find_var_index<'a>(name: &str, parsed_vars: &[&'a str]) -> usize {
    let idx = parsed_vars.iter().enumerate().find(|(_, n)| **n == name);
    match idx {
        Some((i, _)) => i,
        None => {
            panic!("This is probably a bug. I don't know variable {}", name)
        }
    }
}

/// Handles the case that a token is a unary operator and returns a tuple.
/// The first element is a node that is either an expression with a unary operator or a
/// number where the unary operator has been applied to. the second element is the number
/// of tokens that are covered by the unary operator and its argument. Note that a unary
/// operator can be a composition of multiple functions.
fn process_unary<'a, T: Copy + FromStr + Debug, O: Operate<'a, T>>(
    token_idx: usize,
    unary_op: fn(T) -> T,
    repr: &'a str,
    parsed_tokens: &[ParsedToken<'a, T, O>],
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
            let (expr, i_forward) = make_expression::<T, O>(
                &parsed_tokens[token_idx + n_uops + 1..],
                parsed_vars,
                UnaryOpWithReprs {
                    reprs: vec_of_uop_reprs,
                    op: uop,
                },
            )?;
            Ok((DeepNode::Expr(expr), i_forward + n_uops + 1))
        }
        ParsedToken::Var(name) => {
            let expr = DeepEx::new(
                vec![DeepNode::Var((find_var_index(name, &parsed_vars), name))],
                BinOpsWithReprs::new(),
                UnaryOpWithReprs {
                    reprs: vec_of_uop_reprs,
                    op: uop,
                },
            )?;
            Ok((DeepNode::Expr(expr), n_uops + 1))
        }
        ParsedToken::Num(n) => Ok((DeepNode::Num(uop.apply(*n)), n_uops + 1)),
        _ => Err(ExError {
            msg: "Invalid parsed token configuration".to_string(),
        }),
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
pub fn make_expression<'a, T, O>(
    parsed_tokens: &[ParsedToken<'a, T, O>],
    parsed_vars: &[&'a str],
    unary_ops: UnaryOpWithReprs<'a, T>,
) -> ExResult<(DeepEx<'a, T>, usize)>
where
    T: Copy + FromStr + Debug,
    O: Operate<'a, T>,
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
                if idx_tkn > 0 && is_operator_binary(op, &parsed_tokens[idx_tkn - 1]) {
                    bin_ops.push(op.bin()?);
                    reprs_bin_ops.push(op.repr());
                    idx_tkn += 1;
                } else {
                    let (node, idx_forward) = process_unary(
                        idx_tkn,
                        op.unary()?,
                        op.repr(),
                        &parsed_tokens,
                        &parsed_vars,
                    )?;
                    nodes.push(node);
                    idx_tkn += idx_forward;
                }
            }
            ParsedToken::Num(n) => {
                nodes.push(DeepNode::Num(*n));
                idx_tkn += 1;
            }
            ParsedToken::Var(name) => {
                nodes.push(DeepNode::Var((find_var_index(name, &parsed_vars), name)));
                idx_tkn += 1;
            }
            ParsedToken::Paren(p) => match p {
                Paren::Open => {
                    idx_tkn += 1;
                    let (expr, i_forward) = make_expression::<T, O>(
                        &parsed_tokens[idx_tkn..],
                        parsed_vars,
                        UnaryOpWithReprs::new(),
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

pub fn prioritized_indices<T: Copy + Debug>(
    bin_ops: &[BinOp<T>],
    nodes: &[DeepNode<T>],
) -> ExprIdxVec {
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
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprsBuf<T: Copy> {
    pub reprs: SmallVec<[String; N_BINOPS_OF_DEEPEX_ON_STACK]>,
    pub ops: BinOpVec<T>,
}
impl<T: Copy> BinOpsWithReprsBuf<T> {
    pub fn from_deepex<'a>(bin_ops_in: &BinOpsWithReprs<'a, T>) -> Self {
        BinOpsWithReprsBuf {
            reprs: bin_ops_in
                .reprs
                .iter()
                .map(|repr| repr.to_string())
                .collect(),
            ops: bin_ops_in.ops.clone(),
        }
    }
    pub fn to_deepex<'a>(&'a self) -> BinOpsWithReprs<'a, T> {
        BinOpsWithReprs {
            reprs: self.reprs.iter().map(|repr| repr.as_str()).collect(),
            ops: self.ops.clone(),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOpWithReprsBuf<T: Copy> {
    pub reprs: SmallVec<[String; N_UNARYOPS_OF_DEEPEX_ON_STACK]>,
    pub op: UnaryOp<T>,
}
impl<T: Copy> UnaryOpWithReprsBuf<T> {
    pub fn from_deepex<'a>(unary_op_in: &UnaryOpWithReprs<'a, T>) -> Self {
        UnaryOpWithReprsBuf {
            reprs: unary_op_in
                .reprs
                .iter()
                .map(|repr| repr.to_string())
                .collect(),
            op: unary_op_in.op.clone(),
        }
    }
    pub fn to_deepex<'a>(&'a self) -> UnaryOpWithReprs<'a, T> {
        UnaryOpWithReprs {
            reprs: self.reprs.iter().map(|repr| repr.as_str()).collect(),
            op: self.op.clone(),
        }
    }
}
