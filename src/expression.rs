use num::Float;

use crate::{operators::BinOp, util::apply_unary_ops};

#[derive(Debug)]
pub enum Node<T: Float> {
    Expr(Expression<T>),
    Num(T),
    Var(usize),
}
#[derive(Debug)]
pub struct Expression<T: Float> {
    pub nodes: Vec<Node<T>>,
    pub bin_ops: Vec<BinOp<T>>,
    // the last unary operator is applied first to the result
    // of the evaluation of nodes and binary operators
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
