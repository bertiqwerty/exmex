use crate::{
    data_type::DataType,
    definitions::{
        N_BINOPS_OF_DEEPEX_ON_STACK, N_NODES_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK,
        N_VARS_ON_STACK,
    },
    expression::deep_details,
    format_exerr,
    operators::{BinOp, UnaryOp},
    parser, ExError, ExResult, FlatEx, MakeOperators, MatchLiteral, Operator,
};
use num::Float;
use smallvec::{smallvec, SmallVec};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
    iter,
    str::FromStr,
};

use super::{flat::ExprIdxVec, flat_details};

/// Container of binary operators of one expression.
pub type BinOpVec<T> = SmallVec<[BinOp<T>; N_NODES_ON_STACK]>;

pub fn parse<'a, T, F>(
    text: &'a str,
    ops: &[Operator<'a, T>],
    is_numeric: F,
) -> ExResult<DeepEx<'a, T>>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    F: Fn(&'a str) -> Option<&'a str>,
{
    let parsed_tokens = parser::tokenize_and_analyze(text, ops, is_numeric)?;
    parser::check_parsed_token_preconditions(&parsed_tokens)?;
    let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
    let (expr, _) =
        deep_details::make_expression(&parsed_tokens[0..], &parsed_vars, UnaryOpWithReprs::new())?;
    Ok(expr)
}

fn lift_nodes<T: Clone + Debug>(deepex: &mut DeepEx<T>) {
    if deepex.nodes.len() == 1 && deepex.unary_op.op.len() == 0 {
        if let DeepNode::Expr(e) = &deepex.nodes[0] {
            *deepex = (**e).clone();
        }
    } else {
        for node in &mut deepex.nodes {
            if let DeepNode::Expr(e) = node {
                if e.nodes.len() == 1 && e.unary_op.op.len() == 0 {
                    match &mut e.nodes[0] {
                        DeepNode::Num(n) => *node = DeepNode::Num(n.clone()),
                        DeepNode::Var(v) => {
                            *node = DeepNode::Var(*v);
                        }
                        DeepNode::Expr(e_deeper) => {
                            lift_nodes(e_deeper);
                            if e_deeper.nodes.len() == 1 && e_deeper.unary_op.op.len() == 0 {
                                *node = DeepNode::Expr(e_deeper.clone());
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A deep node can be an expression, a number, or
/// a variable.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum DeepNode<'a, T: Clone + Debug> {
    /// Boxing this due to <https://rust-lang.github.io/rust-clippy/master/index.html#large_enum_variant>
    Expr(Box<DeepEx<'a, T>>),
    Num(T),
    /// The contained integer points to the index of the variable in the slice of
    /// variables passed to [`eval`](Expression::eval).
    Var((usize, &'a str)),
}
impl<'a, T: Debug> DeepNode<'a, T>
where
    T: Clone + Debug + Float,
{
    pub fn zero() -> Self {
        DeepNode::Num(T::from(0.0).unwrap())
    }
    pub fn one() -> Self {
        DeepNode::Num(T::from(1.0).unwrap())
    }
    pub fn num(n: T) -> Self {
        DeepNode::Num(n)
    }
}
impl<'a, T: Clone + Debug> Debug for DeepNode<'a, T>
where
    T: Clone + Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DeepNode::Expr(e) => write!(f, "{}", e),
            DeepNode::Num(n) => write!(f, "{:?}", n),
            DeepNode::Var((_, var_name)) => write!(f, "{}", var_name),
        }
    }
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct BinOpsWithReprs<'a, T: Clone> {
    pub reprs: SmallVec<[&'a str; N_BINOPS_OF_DEEPEX_ON_STACK]>,
    pub ops: BinOpVec<T>,
}
impl<'a, T: Clone> BinOpsWithReprs<'a, T> {
    pub fn new() -> Self {
        BinOpsWithReprs {
            reprs: smallvec![],
            ops: BinOpVec::new(),
        }
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
            reprs: smallvec![],
            op: UnaryOp::new(),
        }
    }

    pub fn append_front(&mut self, other: &UnaryOpWithReprs<'a, T>) {
        self.op.append_after(&other.op);
        self.reprs = other
            .reprs
            .iter()
            .chain(self.reprs.iter())
            .copied()
            .collect();
    }
}

/// A deep expression evaluates co-recursively since its nodes can contain other deep
/// expressions.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct DeepEx<'a, T: Clone + Debug> {
    /// Nodes can be numbers, variables, or other expressions.
    nodes: Vec<DeepNode<'a, T>>,
    /// Binary operators applied to the nodes according to their priority.
    pub bin_ops: BinOpsWithReprs<'a, T>,
    /// Unary operators are applied to the result of evaluating all nodes with all
    /// binary operators.
    unary_op: UnaryOpWithReprs<'a, T>,
    var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
}

impl<'a, T> DeepEx<'a, T>
where
    T: Clone + Debug,
{
    pub fn flatten<OF, LMF>(self) -> FlatEx<T, OF, LMF>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LMF: MatchLiteral,
    {
        let (nodes, ops) = deep_details::flatten_vecs(&self, 0);
        let indices = flat_details::prioritized_indices_flat(&ops, &nodes);
        FlatEx::new(
            nodes,
            ops,
            indices,
            self.var_names.iter().map(|s|s.to_string()).collect::<SmallVec<_>>(),
            self.unparse_raw(),
        )
    }
    /// Compiles expression, needed for partial differentation.
    pub fn compile(&mut self) {
        lift_nodes(self);

        let prio_indices = deep_details::prioritized_indices(&self.bin_ops.ops, &self.nodes);
        let mut num_inds = prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();

        let mut already_declined: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec![false; self.nodes.len()];

        for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (DeepNode::Num(num_1), DeepNode::Num(num_2)) = (node_1, node_2) {
                if !(already_declined[num_idx] || already_declined[num_idx + 1]) {
                    let bin_op_result =
                        (self.bin_ops.ops[bin_op_idx].apply)(num_1.clone(), num_2.clone());
                    self.nodes[num_idx] = DeepNode::Num(bin_op_result);
                    self.nodes.remove(num_idx + 1);
                    already_declined.remove(num_idx + 1);
                    // reduce indices after removed position
                    for num_idx_after in num_inds.iter_mut() {
                        if *num_idx_after > num_idx {
                            *num_idx_after -= 1;
                        }
                    }
                    used_prio_indices.push(bin_op_idx);
                }
            } else {
                already_declined[num_idx] = true;
                already_declined[num_idx + 1] = true;
            }
        }

        let mut resulting_reprs = smallvec![];
        self.bin_ops.ops = self
            .bin_ops
            .ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|(i, bin_op)| {
                resulting_reprs.push(self.bin_ops.reprs[i]);
                bin_op.clone()
            })
            .collect();
        self.bin_ops.reprs = resulting_reprs;

        if self.nodes.len() == 1 {
            if let DeepNode::Num(n) = self.nodes[0].clone() {
                self.nodes[0] = DeepNode::Num(self.unary_op.op.apply(n));
                self.unary_op.op.clear();
                self.unary_op.reprs.clear();
            }
        }
    }

    pub fn new(
        nodes: Vec<DeepNode<'a, T>>,
        bin_ops: BinOpsWithReprs<'a, T>,
        unary_op: UnaryOpWithReprs<'a, T>,
    ) -> ExResult<DeepEx<'a, T>> {
        if nodes.len() != bin_ops.ops.len() + 1 {
            Err(format_exerr!(
                "mismatch between number of nodes {:?} and binary operators {:?} ({} vs {})",
                nodes,
                bin_ops.ops,
                nodes.len(),
                bin_ops.ops.len()
            ))
        } else {
            let mut found_vars = SmallVec::<[&str; N_VARS_ON_STACK]>::new();
            for node in &nodes {
                match node {
                    DeepNode::Num(_) => (),
                    DeepNode::Var((_, name)) => {
                        if !found_vars.contains(name) {
                            found_vars.push(name);
                        }
                    }
                    DeepNode::Expr(e) => {
                        for name in &e.var_names {
                            if !found_vars.contains(name) {
                                found_vars.push(name);
                            }
                        }
                    }
                }
            }
            found_vars.sort_unstable();
            let mut expr = DeepEx {
                nodes,
                bin_ops,
                unary_op,
                var_names: found_vars,
            };
            expr.compile();
            Ok(expr)
        }
    }

    pub fn from_node(node: DeepNode<'a, T>) -> DeepEx<'a, T> {
        DeepEx::new(vec![node], BinOpsWithReprs::new(), UnaryOpWithReprs::new()).unwrap()
    }

    pub fn one() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::one())
    }

    pub fn zero() -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::zero())
    }

    pub fn from_num(x: T) -> DeepEx<'a, T>
    where
        T: Float,
    {
        DeepEx::from_node(DeepNode::num(x))
    }

    pub fn with_new_unary_op(self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        Self {
            nodes: self.nodes,
            bin_ops: self.bin_ops,
            var_names: self.var_names,
            unary_op,
        }
    }

    pub fn bin_ops(&self) -> &BinOpsWithReprs<'a, T> {
        &self.bin_ops
    }

    pub fn unary_op(&self) -> &UnaryOpWithReprs<'a, T> {
        &self.unary_op
    }

    pub fn nodes(&self) -> &Vec<DeepNode<'a, T>> {
        &self.nodes
    }

    fn is_num(&self, num: T) -> bool
    where
        T: Float,
    {
        self.nodes.len() == 1
            && match &self.nodes[0] {
                DeepNode::Num(n) => self.unary_op.op.apply(*n) == num,
                DeepNode::Expr(e) => e.is_num(num),
                _ => false,
            }
    }

    pub fn is_one(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(1.0).unwrap())
    }

    pub fn is_zero(&self) -> bool
    where
        T: Float,
    {
        self.is_num(T::from(0.0).unwrap())
    }


    pub fn reset_vars(
        &mut self,
        new_var_names: SmallVec<[&'a str; N_VARS_ON_STACK]>,
    ) {
        for node in &mut self.nodes {
            match node {
                DeepNode::Expr(e) => e.reset_vars(new_var_names.clone()),
                DeepNode::Var((i, var_name)) => {
                    for (new_idx, new_name) in new_var_names.iter().enumerate() {
                        if var_name == new_name {
                            *i = new_idx;
                        }
                    }
                }
                _ => (),
            }
        }
        self.var_names = new_var_names;
    }

    pub fn var_names_union(self, other: Self) -> (Self, Self) {
        let mut all_var_names = self.var_names.clone();
        for name in other.var_names.clone() {
            if !all_var_names.contains(&name) {
                all_var_names.push(name);
            }
        }
        all_var_names.sort_unstable();
        let mut self_vars_updated = self;
        let mut other_vars_updated = other;
        self_vars_updated.reset_vars(all_var_names.clone());
        other_vars_updated.reset_vars(all_var_names);
        (self_vars_updated, other_vars_updated)
    }

    pub fn var_names_like_other(mut self, other: &Self) -> Self {
        self.var_names = other.var_names.clone();
        self
    }

    /// Applies a binary operator to self and other
    pub fn operate_bin(self, other: Self, bin_op: BinOpsWithReprs<'a, T>) -> Self {
        let (self_vars_updated, other_vars_updated) = self.var_names_union(other);
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

    /// Applies a unary operator to self
    pub fn operate_unary(mut self, unary_op: UnaryOpWithReprs<'a, T>) -> Self {
        self.unary_op.append_front(&unary_op);
        self.compile();
        self
    }

    pub fn unparse_raw(&self) -> String {
        let mut node_strings = self.nodes.iter().map(|n| match n {
            DeepNode::Num(n) => format!("{:?}", n),
            DeepNode::Var((_, var_name)) => format!("{{{}}}", var_name),
            DeepNode::Expr(e) => {
                if e.unary_op.op.len() == 0 {
                    format!("({})", e.unparse_raw())
                } else {
                    e.unparse_raw()
                }
            }
        });
        let mut bin_op_strings = self.bin_ops.reprs.iter();
        // a valid expression has at least one node
        let first_node_str = node_strings.next().unwrap();
        let node_with_bin_ops_string = node_strings.fold(first_node_str, |mut res, node_str| {
            let bin_op_str = bin_op_strings.next().unwrap();
            res.push_str(bin_op_str);
            res.push_str(node_str.as_str());
            res
        });
        let unary_op_string = self
            .unary_op
            .reprs
            .iter()
            .fold(String::new(), |mut res, uop_str| {
                res.push_str(uop_str);
                res.push('(');
                res
            });
        let closings = iter::repeat(")").take(self.unary_op.op.len()).fold(
            String::new(),
            |mut res, closing| {
                res.push_str(closing);
                res
            },
        );
        if self.unary_op.op.len() == 0 {
            node_with_bin_ops_string
        } else {
            format!(
                "{}{}{}",
                unary_op_string, node_with_bin_ops_string, closings
            )
        }
    }

    pub fn from_ops(text: &'a str, ops: &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>
    where
        <T as std::str::FromStr>::Err: Debug,
        T: DataType,
    {
        parse(text, ops, parser::is_numeric_text)
    }
}

impl<'a, T: Clone + Debug> Display for DeepEx<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.unparse_raw())
    }
}

#[cfg(test)]
use crate::{
    expression::deep_details::prioritized_indices,
    expression::partial_derivatives::partial_deepex,
    operators::{FloatOpsFactory, VecOfUnaryFuncs},
    util::assert_float_eq_f64,
    Express,
};

#[cfg(test)]
pub fn from_str(text: &str) -> ExResult<DeepEx<f64>> {
    let ops = FloatOpsFactory::<f64>::make();
    DeepEx::from_ops(text, &ops)
}

#[test]
fn test_reset_vars() {
    let deepex = from_str("2*z+x+y * .5").unwrap();
    let ref_vars = ["x", "y", "z"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex.var_names[i], *rv);
    }
    let deepex2 = from_str("a*c*b").unwrap();
    let ref_vars = ["a", "b", "c"];
    for (i, rv) in ref_vars.iter().enumerate() {
        assert_eq!(deepex2.var_names[i], *rv);
    }
    let (deepex_, deepex2_) = deepex.clone().var_names_union(deepex2.clone());
    let all_vars = ["a", "b", "c", "x", "y", "z"];
    for (i, av) in all_vars.iter().enumerate() {
        assert_eq!(deepex_.var_names[i], *av);
        assert_eq!(deepex2_.var_names[i], *av);
    }
    assert_eq!(deepex.unparse_raw(), deepex_.unparse_raw());
    assert_eq!(deepex2.unparse_raw(), deepex2_.unparse_raw());
}

#[test]
fn test_var_name_union() -> ExResult<()> {
    fn test(str_1: &str, str_2: &str, var_names: &[&str]) -> ExResult<()> {
        let first = from_str(str_1)?;
        let second = from_str(str_2)?;
        let (first, second) = first.var_names_union(second);

        assert_eq!(first.var_names.len(), var_names.len());
        assert_eq!(second.var_names.len(), var_names.len());
        for vn in first.var_names {
            assert!(var_names.contains(&vn));
        }
        for vn in second.var_names {
            assert!(var_names.contains(&vn));
        }
        Ok(())
    }

    test("x", "y", &["x", "y"])?;
    test("x+y*z", "z+y", &["x", "y", "z"])?;
    Ok(())
}

#[cfg(test)]
pub fn eval<T>(deepex: &DeepEx<T>, vars: &[T]) -> ExResult<T>
where
    T: Clone + Debug,
{
    let mut numbers = deepex
        .nodes
        .iter()
        .map(|node| -> ExResult<T> {
            match node {
                DeepNode::Num(n) => Ok(n.clone()),
                DeepNode::Var((idx, _)) => Ok(vars[*idx].clone()),
                DeepNode::Expr(e) => eval(e, vars),
            }
        })
        .collect::<ExResult<SmallVec<[T; N_NODES_ON_STACK]>>>()?;
    let mut ignore: SmallVec<[bool; N_NODES_ON_STACK]> = smallvec![false; deepex.nodes.len()];
    let prio_indices = prioritized_indices(&deepex.bin_ops.ops, &deepex.nodes);
    for (i, &bin_op_idx) in prio_indices.iter().enumerate() {
        let num_idx = prio_indices[i];
        let mut shift_left = 0usize;
        while ignore[num_idx - shift_left] {
            shift_left += 1usize;
        }
        let mut shift_right = 1usize;
        while ignore[num_idx + shift_right] {
            shift_right += 1usize;
        }
        let num_1 = numbers[num_idx - shift_left].clone();
        let num_2 = numbers[num_idx + shift_right].clone();
        numbers[num_idx - shift_left] = (deepex.bin_ops.ops[bin_op_idx].apply)(num_1, num_2);
        ignore[num_idx + shift_right] = true;
    }
    Ok(deepex.unary_op.op.apply(numbers[0].clone()))
}

#[test]
fn test_var_names() {
    let deepex = from_str("x+y+{x}+z*(-y)").unwrap();
    let reference: SmallVec<[&str; N_VARS_ON_STACK]> = smallvec!["x", "y", "z"];
    assert_eq!(deepex.var_names, reference);
}

#[test]
fn test_deep_compile() {
    let ops = FloatOpsFactory::make();
    let nodes = vec![DeepNode::Num(4.5), DeepNode::Num(0.5), DeepNode::Num(1.4)];
    let bin_ops = BinOpsWithReprs {
        reprs: smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec![ops[6].unary().unwrap()]),
    };
    let deep_ex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();

    let bin_ops = BinOpsWithReprs {
        reprs: smallvec![ops[1].repr(), ops[3].repr()],
        ops: smallvec![ops[1].bin().unwrap(), ops[3].bin().unwrap()],
    };
    let unary_op = UnaryOpWithReprs {
        reprs: smallvec![ops[6].repr()],
        op: UnaryOp::from_vec(smallvec![ops[6].unary().unwrap()]),
    };
    let nodes = vec![
        DeepNode::Num(4.5),
        DeepNode::Num(0.5),
        DeepNode::Expr(Box::new(deep_ex)),
    ];
    let deepex = DeepEx::new(nodes, bin_ops, unary_op).unwrap();
    assert_eq!(deepex.nodes.len(), 1);
    match deepex.nodes[0] {
        DeepNode::Num(n) => assert_float_eq_f64(deepex.unary_op.op.apply(n), n),
        _ => {
            unreachable!();
        }
    }
}

#[test]
fn test_deep_lift_node() {
    let deepex = from_str("(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))").unwrap();
    println!("{}", deepex);
    assert_eq!(
        format!("{}", deepex),
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))"
    );

    let deepex = from_str("(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "{a}+{x}^2.0*{x}^2.0");

    let deepex = from_str("1+(((a+x^2*x^2)))").unwrap();
    println!("{}", deepex);
    assert_eq!(format!("{}", deepex), "1.0+({a}+{x}^2.0*{x}^2.0)");
    let mut ddeepex = partial_deepex(1, deepex, &FloatOpsFactory::make()).unwrap();
    ddeepex.compile();
    println!("{}", ddeepex);
    assert_eq!(
        format!("{}", ddeepex),
        "(({x}^2.0)*({x}*2.0))+(({x}*2.0)*({x}^2.0))"
    );
}

#[test]
fn test_deep_compile_2() {
    let expr = from_str("1.0 * 3 * 2 * x / 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 2.0);
    let expr = from_str("x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+2+3+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))")
        .unwrap();
    assert_eq!(
        "{x}*0.25+{x}*8.0+5.0+7.0*sin({y})-{z}/sin(1.5/(1.0-{x}*4.0))",
        expr.unparse_raw()
    );
    let expr = from_str("x + 1 - 2").unwrap();
    assert_float_eq_f64(eval(&expr, &[0.0]).unwrap(), -1.0);
    let expr = from_str("x - 1 + 2").unwrap();
    assert_float_eq_f64(eval(&expr, &[0.0]).unwrap(), 1.0);
    let expr = from_str("x * 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 4.0 / 3.0);
    let expr = from_str("x / 2 / 3").unwrap();
    assert_float_eq_f64(eval(&expr, &[2.0]).unwrap(), 1.0 / 3.0);
}

#[test]
fn test_operate_unary() -> ExResult<()> {
    let lstr = "x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)+x+y+x+z*(-y)";
    let deepex = from_str(lstr)?;
    let mut funcs = VecOfUnaryFuncs::new();
    funcs.push(|x: f64| x * 1.23456);
    let deepex = deepex.operate_unary(UnaryOpWithReprs {
        reprs: smallvec!["eagle"],
        op: UnaryOp::from_vec(funcs),
    });
    assert_float_eq_f64(
        eval(&deepex, &[1.0, 1.75, 2.25])?,
        -0.23148000000000002 * 8.0,
    );
    Ok(())
}

#[test]
fn test_unparse() -> ExResult<()> {
    fn test(text: &str, text_ref: &str) -> ExResult<()> {
        let flatex = FlatEx::<f64>::from_str(text)?;
        assert_eq!(flatex.unparse(), text);

        let deepex = DeepEx::<f64>::from_ops(text, &FloatOpsFactory::make())?;
        assert_eq!(deepex.unparse_raw(), text_ref);
        Ok(())
    }
    let text = "5+x";
    let text_ref = "5.0+{x}";
    test(text, text_ref)?;
    let text = "sin(5+var)^(1/{y})+{var}";
    let text_ref = "sin(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref)?;
    let text = "-(5+var)^(1/{y})+{var}";
    let text_ref = "-(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-(5+var)^(1/{y})))+{var}";
    let text_ref = "cos(sin(-(5.0+{var})^(1.0/{y})))+{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-5+var^(1/{y})))-{var}";
    let text_ref = "cos(sin(-5.0+{var}^(1.0/{y})))-{var}";
    test(text, text_ref)?;
    let text = "cos(sin(-z+var*(1/{y})))+{var}";
    let text_ref = "cos(sin(-({z})+{var}*(1.0/{y})))+{var}";
    test(text, text_ref)?;
    Ok(())
}
