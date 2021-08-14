use crate::{
    definitions::N_NODES_ON_STACK,
    expression::deep::{DeepEx, DeepNode, ExprIdxVec},
    operators::UnaryOp,
    BinOp, ExParseError,
};
use smallvec::{smallvec, SmallVec};
use std::fmt::{self, Debug, Display, Formatter};

pub type FlatNodeVec<T> = SmallVec<[FlatNode<T>; N_NODES_ON_STACK]>;
pub type FlatOpVec<T> = SmallVec<[FlatOp<T>; N_NODES_ON_STACK]>;

/// A `FlatOp` contains besides a binary operation an optional unary operation that
/// will be executed after the binary operation in case of its existence.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatOp<T: Copy> {
    unary_op: UnaryOp<T>,
    bin_op: BinOp<T>,
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub enum FlatNodeKind<T: Copy> {
    Num(T),
    Var(usize),
}

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatNode<T: Copy> {
    kind: FlatNodeKind<T>,
    unary_op: UnaryOp<T>,
}

impl<T: Copy> FlatNode<T> {
    pub fn from_kind(kind: FlatNodeKind<T>) -> FlatNode<T> {
        return FlatNode {
            kind: kind,
            unary_op: UnaryOp::new(),
        };
    }
}

fn flatten_vecs<T: Copy + Debug>(
    deep_expr: &DeepEx<T>,
    prio_offset: i32,
) -> (FlatNodeVec<T>, FlatOpVec<T>) {
    let mut flat_nodes = FlatNodeVec::<T>::new();
    let mut flat_ops = FlatOpVec::<T>::new();

    for (node_idx, node) in deep_expr.nodes().iter().enumerate() {
        match node {
            DeepNode::Num(num) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Num(*num));
                flat_nodes.push(flat_node);
            }
            DeepNode::Var((idx, _)) => {
                let flat_node = FlatNode::from_kind(FlatNodeKind::Var(*idx));
                flat_nodes.push(flat_node);
            }
            DeepNode::Expr(e) => {
                let (mut sub_nodes, mut sub_ops) = flatten_vecs(e, prio_offset + 100i32);
                flat_nodes.append(&mut sub_nodes);
                flat_ops.append(&mut sub_ops);
            }
        };
        if node_idx < deep_expr.bin_ops().ops.len() {
            let prio_adapted_bin_op = BinOp {
                apply: deep_expr.bin_ops().ops[node_idx].apply,
                prio: deep_expr.bin_ops().ops[node_idx].prio + prio_offset,
            };
            flat_ops.push(FlatOp {
                bin_op: prio_adapted_bin_op,
                unary_op: UnaryOp::new(),
            });
        }
    }

    if deep_expr.unary_op().op.len() > 0 {
        if flat_ops.len() > 0 {
            // find the last binary operator with the lowest priority of this expression,
            // since this will be executed as the last one
            let low_prio_op = match flat_ops.iter_mut().rev().min_by_key(|op| op.bin_op.prio) {
                None => panic!("cannot have more than one flat node but no binary ops"),
                Some(x) => x,
            };
            low_prio_op
                .unary_op
                .append_front(&mut deep_expr.unary_op().op.clone());
        } else {
            flat_nodes[0]
                .unary_op
                .append_front(&mut deep_expr.unary_op().op.clone());
        }
    }
    (flat_nodes, flat_ops)
}

fn prioritized_indices_flat<T: Copy>(ops: &[FlatOp<T>], nodes: &FlatNodeVec<T>) -> ExprIdxVec {
    let prio_increase =
        |bin_op_idx: usize| match (&nodes[bin_op_idx].kind, &nodes[bin_op_idx + 1].kind) {
            (FlatNodeKind::Num(_), FlatNodeKind::Num(_)) => {
                let prio_inc = 5;
                &ops[bin_op_idx].bin_op.prio * 10 + prio_inc
            }
            _ => &ops[bin_op_idx].bin_op.prio * 10,
        };
    let mut indices: ExprIdxVec = (0..ops.len()).collect();
    indices.sort_by(|i1, i2| {
        let prio_i1 = prio_increase(*i1);
        let prio_i2 = prio_increase(*i2);
        prio_i2.partial_cmp(&prio_i1).unwrap()
    });
    indices
}

/// Flattens a deep expression
/// The result does not contain any recursive structures and is faster to evaluate.
pub fn flatten<T: Copy + Debug>(deepex: DeepEx<T>) -> FlatEx<T> {
    let (nodes, ops) = flatten_vecs(&deepex, 0);
    let indices = prioritized_indices_flat(&ops, &nodes);
    let n_unique_vars = deepex.n_vars();
    FlatEx {
        nodes: nodes,
        ops: ops,
        prio_indices: indices,
        n_unique_vars: n_unique_vars,
        deepex: Some(deepex),
    }
}

/// This is the core data type representing a flattened expression and the result of
/// parsing a string. We use flattened expressions to make efficient evaluation possible.
/// Simplified, a flat expression consists of a [`SmallVec`](SmallVec) of nodes and a
/// [`SmallVec`](SmallVec) of operators that are applied to the nodes in an order following
/// operator priorities.
///
/// You create an expression with the `parse` function or one of its
/// variants, namely `parse_with_default_ops` and `parse_with_number_pattern`.
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::{parse_with_default_ops};
///
/// // create an expression by parsing a string
/// let expr = parse_with_default_ops::<f32>("sin(1+y)*x")?;
/// assert!((expr.eval(&[2.0, 1.5])? - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The second argument `&[2.0, 1.5]` in the call of [`eval`](FlatEx::eval) specifies the
/// variable values in the order of their occurrence in the string.
/// In this example, we want to evaluate the expression for the varibale values `y=2.0` and `x=1.5`.
/// Variables in the string to-be-parsed are all substrings that are no numbers, no
/// operators, and no parentheses.
///
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatEx<'a, T: Copy + Debug> {
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
    deepex: Option<DeepEx<'a, T>>,
}

impl<'a, T: Copy + Debug> FlatEx<'a, T> {
    /// Evaluates an expression with the given variable values and returns the computed
    /// result.
    ///
    /// # Arguments
    ///
    /// * `vars` - Values of the variables of the expression; the n-th value corresponds to
    ///            the n-th variable as given in the string that has been parsed to this expression.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///
    /// # Errors
    ///
    /// If the number of variables in the parsed expression are different from the length of
    /// the variable slice, we return an [`ExParseError`](ExParseError).
    ///
    pub fn eval(&self, vars: &[T]) -> Result<T, ExParseError> {
        if self.n_unique_vars != vars.len() {
            return Err(ExParseError {
                msg: format!(
                    "parsed expression contains {} vars but passed slice has {} elements",
                    self.n_unique_vars,
                    vars.len()
                ),
            });
        }
        let mut numbers = self
            .nodes
            .iter()
            .map(|node| {
                node.unary_op.apply(match node.kind {
                    FlatNodeKind::Num(n) => n,
                    FlatNodeKind::Var(idx) => vars[idx],
                })
            })
            .collect::<SmallVec<[T; 32]>>();
        let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> = smallvec![false; self.nodes.len()];
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = self.prio_indices[i];
            let mut shift_left = 0usize;
            while ignore[num_idx - shift_left] {
                shift_left += 1usize;
            }
            let mut shift_right = 1usize;
            while ignore[num_idx + shift_right] {
                shift_right += 1usize;
            }
            let num_1 = numbers[num_idx - shift_left];
            let num_2 = numbers[num_idx + shift_right];
            numbers[num_idx - shift_left] = {
                let bop_res = (self.ops[bin_op_idx].bin_op.apply)(num_1, num_2);
                self.ops[bin_op_idx].unary_op.apply(bop_res)
            };
            ignore[num_idx + shift_right] = true;
        }
        Ok(numbers[0])
    }

    /// Creates an expression string that corresponds to the `FlatEx` instance. This is
    /// not necessarily the input string. More precisely,
    /// * variable names are forgotten,
    /// * variables are put into curly braces, and
    /// * expressions will be put between parentheses, e.g.,
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::parse_with_default_ops;
    /// let flatex = parse_with_default_ops::<f64>("--sin(z)")?;
    /// assert_eq!(format!("{}", flatex), "-(-(sin({x0})))");
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    ///
    pub fn unparse(&self) -> Result<String, ExParseError> {
        match &self.deepex {
            Some(deepex) => Ok(deepex.unparse()),
            None => Err(ExParseError {
                msg: "unparse impossible, since deep expression optimized away".to_string(),
            }),
        }
    }
    /// Usually, a `FlatEx` instance keeps a nested, deep structure of the expression
    /// that is not necessary for evaluation. This functions removes the deep expression
    /// to reduce memory consumption. [`unparse`](FlatEx::unparse) and the
    /// [`Display`](std::fmt::Display) implementation will stop working after calling this function.
    pub fn clear_deepex(&mut self) {
        self.deepex = None;
    }
}

/// The expression is displayed as a string created by [`unparse`](FlatEx::unparse).
impl<'a, T: Copy + Debug> Display for FlatEx<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unparsed = self.unparse();
        match unparsed {
            Err(e) => write!(f, "{}", e.msg),
            Ok(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(test)]
use crate::{expression::deep::UnaryOpWithReprs, operators::VecOfUnaryFuncs};

#[test]
fn test_operate_unary() {
    let lstr = "x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)";
    let deepex = DeepEx::<f64>::from_str(lstr).unwrap();
    let mut funcs = VecOfUnaryFuncs::new();
    funcs.push(|x: f64| x * 1.23456);
    let deepex = deepex.operate_unary(UnaryOpWithReprs {
        reprs: vec!["eagle"],
        op: UnaryOp::from_vec(funcs),
    });
    let flatex = flatten(deepex);
    assert_float_eq_f64(
        flatex.eval(&[1.0, 1.75, 2.25]).unwrap(),
        -0.23148000000000002 * 8.0,
    );
}

#[cfg(test)]
use crate::{parse_with_default_ops, util::assert_float_eq_f64};

#[test]
fn test_flat_clear() {
    let mut flatex = parse_with_default_ops::<f64>("x*(2*(2*(2*4*8)))").unwrap();
    assert!(flatex.deepex.is_some());
    flatex.clear_deepex();
    assert!(flatex.deepex.is_none());
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flatex.nodes.len(), 2);
}
#[test]
fn test_flat_compile() {
    let flatex = parse_with_default_ops::<f64>("1*sin(2-0.1)").unwrap();
    assert_float_eq_f64(flatex.eval(&[]).unwrap(), 1.9f64.sin());
    assert_eq!(flatex.nodes.len(), 1);

    let flatex = parse_with_default_ops::<f64>("x*(2*(2*(2*4*8)))").unwrap();
    assert_float_eq_f64(flatex.eval(&[1.0]).unwrap(), 2.0 * 2.0 * 2.0 * 4.0 * 8.0);
    assert_eq!(flatex.nodes.len(), 2);

    let flatex = parse_with_default_ops::<f64>("1*sin(2-0.1) + x").unwrap();
    assert_float_eq_f64(flatex.eval(&[0.0]).unwrap(), 1.9f64.sin());
    assert_eq!(flatex.nodes.len(), 2);
    match flatex.nodes[0].kind {
        FlatNodeKind::Num(n) => assert_float_eq_f64(n, 1.9f64.sin()),
        _ => assert!(false),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => assert!(false),
    }

    let flatex = parse_with_default_ops::<f64>("y + 1 - cos(1/(1*sin(2-0.1))-2) + 2 + x").unwrap();
    assert_eq!(flatex.nodes.len(), 3);
    match flatex.nodes[0].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => assert!(false),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Num(_) => (),
        _ => assert!(false),
    }
    match flatex.nodes[2].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 1),
        _ => assert!(false),
    }
}

#[test]
fn test_operator_overloading() {
    fn from_str(text: &str) -> DeepEx<f64> {
        DeepEx::from_str(text).unwrap()
    }
    fn eval<'a>(deepex: &DeepEx<'a, f64>, vars: &[f64], val: f64) {
        assert_float_eq_f64(flatten(deepex.clone()).eval(vars).unwrap(), val);
    }

    let one = from_str("1");
    let two = one.clone() + one.clone();
    eval(&two, &[], 2.0);

    let x_squared = from_str("x*x");
    let two_x_squared = two.clone() * x_squared.clone();
    eval(&two_x_squared, &[0.0], 0.0);
    eval(&two_x_squared, &[1.0], 2.0);
    eval(&two_x_squared, &[2.0], 8.0);
    eval(&two_x_squared, &[3.0], 18.0);
    let some_expr = from_str("x") + from_str("x") * from_str("2") / from_str("x^(.5)");
    eval(&some_expr, &[4.0], 8.0);

    let x_plus_y_plus_z = from_str("x+y+z");
    let y_minus_z = from_str("y-z");
    let prod_of_above = x_plus_y_plus_z.clone() * y_minus_z.clone();
    eval(&prod_of_above, &[1.0, 4.0, 8.0], -52.0);
    let div_of_above = x_plus_y_plus_z.clone() / y_minus_z.clone();
    eval(&div_of_above, &[1.0, 4.0, 8.0], -3.25);
    let sub_of_above = x_plus_y_plus_z.clone() - y_minus_z.clone();
    eval(&sub_of_above, &[1.0, 4.0, 8.0], 17.0);
    let add_of_above = x_plus_y_plus_z + y_minus_z.clone();
    eval(&add_of_above, &[1.0, 4.0, 8.0], 9.0);
    let x_plus_cossin_y_plus_z = from_str("x+cos(sin(y+z))");
    let prod_of_above = x_plus_cossin_y_plus_z * y_minus_z;
    eval(&prod_of_above, &[1.0, 4.0, 8.0], -7.4378625090980925);
}
#[test]
fn test_display() {
    let mut flatex = flatten(DeepEx::<f64>::from_str("sin(var)/5").unwrap());
    assert_eq!(format!("{}", flatex), "sin({x0})/5.0");
    flatex.clear_deepex();
    assert_eq!(
        format!("{}", flatex),
        "unparse impossible, since deep expression optimized away"
    );
}

#[test]
fn test_unparse() {
    fn test(text: &str, text_ref: &str) {
        let flatex = flatten(DeepEx::<f64>::from_str(text).unwrap());
        let deepex = flatex.deepex.unwrap();

        assert_eq!(deepex.unparse(), text_ref);
        let mut flatex_reparsed = flatten(DeepEx::<f64>::from_str(text).unwrap());
        assert_eq!(flatex_reparsed.unparse().unwrap(), text_ref);
        flatex_reparsed.clear_deepex();
        assert!(flatex_reparsed.unparse().is_err());
    }
    let text = "5+x";
    let text_ref = "5.0+{x0}";
    test(text, text_ref);
    let text = "sin(5+var)^(1/{y})+{var}";
    let text_ref = "sin(5.0+{x0})^(1.0/{x1})+{x0}";
    test(text, text_ref);
    let text = "-(5+var)^(1/{y})+{var}";
    let text_ref = "-(5.0+{x0})^(1.0/{x1})+{x0}";
    test(text, text_ref);
    let text = "cos(sin(-(5+var)^(1/{y})))+{var}";
    let text_ref = "cos(sin(-(5.0+{x0})^(1.0/{x1})))+{x0}";
    test(text, text_ref);
    let text = "cos(sin(-5+var^(1/{y})))-{var}";
    let text_ref = "cos(sin(-5.0+{x0}^(1.0/{x1})))-{x0}";
    test(text, text_ref);
    let text = "cos(sin(-z+var*(1/{y})))+{var}";
    let text_ref = "cos(sin(-({x0})+{x1}*(1.0/{x2})))+{x1}";
    test(text, text_ref);
}
