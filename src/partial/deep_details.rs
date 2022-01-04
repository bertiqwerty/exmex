use crate::{
    definitions::{N_BINOPS_OF_DEEPEX_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK},
    expression::flat::{ExprIdxVec, FlatNode, FlatNodeKind, FlatNodeVec, FlatOp, FlatOpVec},
    operators::{BinOp, UnaryOp, VecOfUnaryFuncs},
    parser::{self, Paren, ParsedToken},
    partial::deep::{BinOpVec, BinOpsWithReprs, DeepEx, DeepNode, UnaryOpWithReprs},
    ExError, ExResult,
};
use std::{fmt::Debug, iter, str::FromStr};

use smallvec::SmallVec;

/// Handles the case that a token is a unary operator and returns a tuple.
/// The first element is a node that is either an expression with a unary operator or a
/// number where the unary operator has been applied to. the second element is the number
/// of tokens that are covered by the unary operator and its argument. Note that a unary
/// operator can be a composition of multiple functions.
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
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprsBuf<T: Clone> {
    pub reprs: SmallVec<[String; N_BINOPS_OF_DEEPEX_ON_STACK]>,
    pub ops: BinOpVec<T>,
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOpWithReprsBuf<T> {
    pub reprs: SmallVec<[String; N_UNARYOPS_OF_DEEPEX_ON_STACK]>,
    pub op: UnaryOp<T>,
}

pub fn flatten_vecs<T: Clone + Debug>(
    deep_expr: &DeepEx<T>,
    prio_offset: i64,
) -> (FlatNodeVec<T>, FlatOpVec<T>) {
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    for (node_idx, node) in deep_expr.nodes().iter().enumerate() {
        match node {
            DeepNode::Num(num) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Num(num.clone()));
                flat_nodes.push(flat_node);
            }
            DeepNode::Var((idx, _)) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Var(*idx));
                flat_nodes.push(flat_node);
            }
            DeepNode::Expr(e) => {
                let (mut sub_nodes, mut sub_ops) = flatten_vecs(e, prio_offset + 100i64);
                flat_nodes.append(&mut sub_nodes);
                flat_ops.append(&mut sub_ops);
            }
        };
        if node_idx < deep_expr.bin_ops().ops.len() {
            let prio_adapted_bin_op = BinOp {
                apply: deep_expr.bin_ops().ops[node_idx].apply,
                prio: deep_expr.bin_ops().ops[node_idx].prio + prio_offset,
                is_commutative: deep_expr.bin_ops().ops[node_idx].is_commutative,
            };
            flat_ops.push(FlatOp {
                bin_op: prio_adapted_bin_op,
                unary_op: UnaryOp::new(),
            });
        }
    }

    if deep_expr.unary_op().op.len() > 0 {
        if !flat_ops.is_empty() {
            // find the last binary operator with the lowest priority of this expression,
            // since this will be executed as the last one
            let low_prio_op = match flat_ops.iter_mut().rev().min_by_key(|op| op.bin_op.prio) {
                None => panic!("cannot have more than one flat node but no binary ops"),
                Some(x) => x,
            };
            low_prio_op
                .unary_op
                .append_after(&deep_expr.unary_op().op.clone());
        } else {
            flat_nodes[0]
                .unary_op
                .append_after(&deep_expr.unary_op().op.clone());
        }
    }
    (flat_nodes, flat_ops)
}
