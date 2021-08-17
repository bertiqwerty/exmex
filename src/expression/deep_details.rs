use crate::{
    definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK},
    expression::deep::{BinOpVec, BinOpsWithReprs, DeepEx, DeepNode, ExprIdxVec, UnaryOpWithReprs},
    operators::{BinOp, Operator, UnaryOp, VecOfUnaryFuncs},
    parser::{ExParseError, Paren, ParsedToken},
};
use std::{fmt::Debug, iter, str::FromStr};

use smallvec::SmallVec;

pub const ADD_REPR: &str = "+";
pub const SUB_REPR: &str = "-";
pub const MUL_REPR: &str = "*";
pub const DIV_REPR: &str = "/";

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

pub fn find_overloaded_ops<'a, T: Copy>(all_ops: &[Operator<T>]) -> Option<OverloadedOps<'a, T>> {
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

    Some(OverloadedOps {
        add: find_op(ADD_REPR)?,
        sub: find_op(SUB_REPR)?,
        mul: find_op(MUL_REPR)?,
        div: find_op(DIV_REPR)?,
    })
}

pub fn parsed_tokens_to_deepex<'a, T: Copy + FromStr + Debug>(
    parsed_tokens: &[ParsedToken<'a, T>],
) -> Result<DeepEx<'a, T>, ExParseError> {
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
        .collect::<SmallVec<[_; N_NODES_ON_STACK]>>();
    parsed_vars.sort_unstable();
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
pub fn make_expression<'a, T>(
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
        let iter_of_uops = iter::once((repr, uo)).chain(
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
                        parsed_vars,
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
                        parsed_vars,
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
