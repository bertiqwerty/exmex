use std::fmt::Debug;

use crate::{operators::BinOp, util::apply_unary_ops, ExParseError};

/// Nodes are inputs for binary operators. A node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Node<T: Copy> {
    Expr(Expression<T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var(usize),
}

/// Core data type and the result of parsing a string.
///
/// Usually, you would create an expression with the `parse` function or one of its
/// variants, namely `parse_with_default_ops` and `parse_with_number_pattern`.
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{parse_with_default_ops};
///
/// // create an expression by parsing a string
/// let expr_parsed = parse_with_default_ops::<f32>("sin(1+{x})*{y}")?;
/// let result_parsed = expr_parsed.eval(&[2.0, 1.5]);
/// assert!((result_parsed - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The second argument `&[2.0, 1.5]` in the call of [`eval`](Expression::eval) specifies
/// that we want to evaluate the expression for the varibale values `x=2.0` and `y=1.5`.
/// Note that variables need to be within curly brackets in the string to-be-parsed.
///
/// You can also create the expression directly. In this case you have to make sure that
/// you have `n+1` nodes for `n` binary operators. This can also be evaluated with
/// [`eval`](Expression::eval).
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{BinOp, Expression, Node};
/// // create an expression directly
/// let expr_directly = Expression::new(
///     vec![Node::Num(1.0), Node::Var(0)],
///     vec![
///         BinOp {
///             op: |a: f32, b: f32| a + b,
///             prio: 0
///         }
///     ],
///     vec![|a: f32| a.sin()]
/// )?;
/// let result_directly = expr_directly.eval(&[2.0]);
/// assert!((result_directly - (1.0 + 2.0 as f32).sin()).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Expression<T: Copy> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<Node<T>>,
    /// Binary operators applied to the nodes according to their priority.
    bin_ops: Vec<BinOp<T>>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators. The last unary operator is applied first to the result
    /// of the evaluation of nodes and binary operators
    unary_ops: Vec<fn(T) -> T>,
}

fn prioritized_indices<T: Copy>(bin_ops: &Vec<BinOp<T>>) -> Vec<usize> {
    let mut indices: Vec<_> = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| bin_ops[*i2].prio.partial_cmp(&bin_ops[*i1].prio).unwrap());
    indices
}

impl<T: Copy + Debug> Expression<T> {
    /// Evaluates an expression with the given variable values and returns the computed
    /// result.
    ///
    /// The binary operators of the expression are
    /// applied to the expression's nodes. The order in the `nodes`-vector determines
    /// for which binary operator a node is used as input. More precisely, let us assume
    /// the binary operator with index `i` has the highest priority. Then, the
    /// nodes with index `i` and `i+1` are used as its input. After the binary operator with
    /// the highest priority is evaluated, the result is put into
    /// a the mutable node with index `i`, the number of nodes an operators is reduced by 1
    /// and the operator with the next highest priority is considered, etc. Eventually,
    /// the unary operators map the result of the
    /// evaluation of the binary operators to the final value of the expression.
    ///
    /// # Arguments
    ///
    /// * `expr` - expression to be evaluated
    /// * `vars` - values of the variables of the expression, the n-th value corresponds to
    ///            the n-th variable as given in the string that has been parsed to this expression.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///
    pub fn eval(&self, vars: &[T]) -> T {
        let indices = prioritized_indices(&self.bin_ops);
        let mut numbers = self
            .nodes
            .iter()
            .map(|n| match n {
                Node::Expr(e) => e.eval(&vars),
                Node::Num(n) => *n,
                Node::Var(idx) => vars[*idx],
            })
            .collect::<Vec<T>>();
        let mut num_inds = indices.clone();
        for (i, &bin_op_idx) in indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let num_1 = numbers[num_idx];
            let num_2 = numbers[num_idx + 1];
            numbers[num_idx] = (self.bin_ops[bin_op_idx].op)(num_1, num_2);
            numbers.remove(num_idx + 1);
            // reduce indices after removed position
            for num_idx_after in num_inds.iter_mut() {
                if *num_idx_after > num_idx {
                    *num_idx_after = *num_idx_after - 1;
                }
            }
        }
        apply_unary_ops(&self.unary_ops, numbers[0])
    }

    /// Creates a flat expression, i.e., without any kind of recursion, and checks
    /// whether the number of nodes is by one larger than the number of binary
    /// operators.
    pub fn new(
        nodes: Vec<Node<T>>,
        bin_ops: Vec<BinOp<T>>,
        unary_ops: Vec<fn(T) -> T>,
    ) -> Result<Expression<T>, ExParseError> {
        if nodes.len() != bin_ops.len() + 1 {
            Err(ExParseError {
                msg: "mismatch between number of nodes and binary operators".to_string(),
            })
        } else {
            Ok(Expression {
                nodes: nodes,
                bin_ops: bin_ops,
                unary_ops: unary_ops,
            })
        }
    }
}

#[cfg(test)]
mod test {
    use crate::expression::{prioritized_indices, BinOp};

    #[test]
    fn test_prio() {
        assert_eq!(
            prioritized_indices(&vec![
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                }
            ]),
            vec![1, 0]
        );
        assert_eq!(
            prioritized_indices(&vec![
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                }
            ]),
            vec![1, 3, 0, 2]
        );
        assert_eq!(
            prioritized_indices(&vec![
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                }
            ]),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            prioritized_indices(&vec![
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 2
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 3
                }
            ]),
            vec![3, 2, 1, 0]
        );
        assert_eq!(
            prioritized_indices(&vec![
                BinOp {
                    op: |_, _| 0.0,
                    prio: 3
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 2
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 1
                },
                BinOp {
                    op: |_, _| 0.0,
                    prio: 0
                }
            ]),
            vec![0, 1, 2, 3]
        );
    }
}
