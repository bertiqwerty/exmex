use crate::{operators::BinOp, util::apply_unary_ops};
use num::Float;

/// Nodes are inputs for binary operators. A node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum Node<T: Float> {
    Expr(Expression<T>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval_expr`](eval_expr).
    Var(usize),
}

/// Core data type and the result of parsing a string.
///
/// Usually, you would create an expression with the `parse` function.
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exexpress::{eval_expr, parse_with_default_ops};
///
/// // create an expression by parsing a string
/// let expr_parsed = parse_with_default_ops::<f32>("sin(1+{x})")?;
/// let result_parsed = eval_expr::<f32>(&expr_parsed, &[2.0]);
/// assert!((result_parsed - (1.0 + 2.0 as f32).sin()).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The second argument &[2.0] in the call of `eval_expr` specifies the we want to
/// evaluate the expression for the value 2.0 of our only variable `{x}`. Variables need
/// to be within curly brackets in the string to-be-parsed.
///
/// You can also create the expression directly. In this case you have to make sure that
/// you have `n+1` nodes for `n` binary operators. The binary operators are
/// applied to the nodes. The order in the `nodes`-vector determines
/// for which binary operator a node is used as input. More precisely,
/// nodes `i` and `i+1` are the input of the binary operator `i` with the highest
/// priority. After the calculation with the highest priority, the result is put into
/// a node, the number of nodes an operators is reduced by 1 and the operator with
/// the next highest priority is considered, etc.
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exexpress::{eval_expr, BinOp, Expression, Node};
/// // create an expression directly
/// let expr_directly = Expression {
///     bin_ops: vec![
///         BinOp {
///             op: |a: f32, b: f32| a + b,
///             prio: 0
///         }
///     ],
///     nodes: vec![Node::Num(1.0), Node::Var(0)],
///     unary_ops: vec![|a: f32| a.sin()]
/// };
/// let result_directly = eval_expr::<f32>(&expr_directly, &[2.0]);
/// assert!((result_directly - (1.0 + 2.0 as f32).sin()).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct Expression<T: Float> {
    /// Nodes can be numbers, variables, or other expressions.
    pub nodes: Vec<Node<T>>,
    /// Binary operators applied to the nodes according to their priority.
    pub bin_ops: Vec<BinOp<T>>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators. The last unary operator is applied first to the result
    /// of the evaluation of nodes and binary operators
    pub unary_ops: Vec<fn(T) -> T>,
}

fn prioritized_indices<T: Float>(bin_ops: &Vec<BinOp<T>>) -> Vec<usize> {
    let mut indices: Vec<_> = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| bin_ops[*i2].prio.partial_cmp(&bin_ops[*i1].prio).unwrap());
    indices
}

/// Evaluates an expression with the given variable values and returns the computed result.
///
/// # Arguments
///
/// * `expr` - expression to be evaluated
/// * `vars` - values of the variables of the expression, the n-th value corresponds to
///            the n-th variable as given in the string that has been parsed to this expression.
///            Thereby, only the first occurrence of the variable in the string is relevant.
///
pub fn eval_expr<T: Float + std::fmt::Debug>(expr: &Expression<T>, vars: &[T]) -> T {
    let indices = prioritized_indices(&expr.bin_ops);
    let mut numbers = expr
        .nodes
        .iter()
        .map(|n| match n {
            Node::Expr(e) => eval_expr(e, &vars),
            Node::Num(n) => *n,
            Node::Var(idx) => vars[*idx],
        })
        .collect::<Vec<T>>();
    let mut num_inds = indices.clone();
    for (i, &bin_op_idx) in indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let num_1 = numbers[num_idx];
        let num_2 = numbers[num_idx + 1];
        numbers[num_idx] = (expr.bin_ops[bin_op_idx].op)(num_1, num_2);
        numbers.remove(num_idx + 1);
        // reduce indices after removed position
        for num_idx_after in num_inds.iter_mut() {
            if *num_idx_after > num_idx {
                *num_idx_after = *num_idx_after - 1;
            }
        }
    }
    apply_unary_ops(&expr.unary_ops, numbers[0])
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
    }
}
