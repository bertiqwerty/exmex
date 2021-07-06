use num::Float;

use crate::util::apply_unary_ops;

#[derive(Debug)]
pub enum Node<T: Float> {
    Expr(Expression<T>),
    Num(T),
}
#[derive(Debug)]
pub struct Expression<T: Float> {
    pub nodes: Vec<Node<T>>,
    pub bin_ops: Vec<BinOp<T>>,
    // the last unary operator is applied first to the result
    // of the evaluation of nodes and binary operators
    pub unary_ops: Vec<fn(T) -> T>  
}


#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct BinOp<T: Copy> {
    pub op: fn(T, T) -> T,
    pub prio: i16,
}

fn priorized_indices<T: Float>(bin_ops: &Vec<BinOp<T>>) -> Vec<usize> {
    let mut indices: Vec<_> = (0..bin_ops.len()).collect();
    indices.sort_by(|i1, i2| {
        bin_ops[*i2]
            .prio
            .partial_cmp(&bin_ops[*i1].prio)
            .unwrap()
    });
    indices
}

pub fn eval_expr<T: Float + std::fmt::Debug>(exp: &Expression<T>) -> T {
    let indices = priorized_indices(&exp.bin_ops);
    let mut numbers = exp
        .nodes
        .iter()
        .map(|n| match n {
            Node::Expr(e) => eval_expr(e),
            Node::Num(n) => *n,
        })
        .collect::<Vec<T>>();
    let mut num_inds = indices.clone();
    for (i, &bin_op_idx) in indices.iter().enumerate() {
        let num_idx = num_inds[i];
        let num_1 = numbers[num_idx];
        let num_2 = numbers[num_idx + 1];
        numbers[num_idx] = (exp.bin_ops[bin_op_idx].op)(num_1, num_2);
        numbers.remove(num_idx + 1);
        // reduce indices after removed position
        for num_idx_after in num_inds.iter_mut() {
            if *num_idx_after > num_idx {
                *num_idx_after = *num_idx_after - 1;
            }
        }
    }
    apply_unary_ops(&exp.unary_ops, numbers[0])
}

#[cfg(test)]
mod test {
    use crate::expression::{BinOp, priorized_indices};

    #[test]
    fn test_prio() {
        assert_eq!(
            priorized_indices(&vec![
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
            priorized_indices(&vec![
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
            priorized_indices(&vec![
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
