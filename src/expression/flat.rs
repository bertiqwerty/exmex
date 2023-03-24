use self::detail::{FlatNode, FlatNodeKind, FlatNodeVec, FlatOpVec};
use crate::data_type::DataType;
use crate::definitions::{N_NODES_ON_STACK, N_VARS_ON_STACK};
use crate::expression::{
    deep::{DeepEx, DeepNode},
    Express,
};
use crate::operators::UnaryOp;
use crate::{
    format_exerr, BinOp, Calculate, ExError, ExResult, FloatOpsFactory, MakeOperators,
    MatchLiteral, NumberMatcher,
};

use smallvec::SmallVec;
use std::fmt::{self, Debug, Display, Formatter};
use std::iter;
use std::marker::PhantomData;
use std::str::FromStr;

const DEPTH_PRIO_STEP: i64 = 1000;
pub type ExprIdxVec = SmallVec<[usize; N_NODES_ON_STACK]>;
mod detail {
    use std::{fmt::Debug, marker::PhantomData, str::FromStr};

    use smallvec::{smallvec, SmallVec};
    use std::mem;

    use crate::{
        data_type::DataType,
        definitions::{N_NODES_ON_STACK, N_UNARYOPS_OF_DEEPEX_ON_STACK, N_VARS_ON_STACK},
        expression::{eval_binary, number_tracker::NumberTracker},
        format_exerr,
        operators::{OperateBinary, UnaryOp},
        parser::{self, Paren, ParsedToken},
        BinOp, ExError, ExResult, FlatEx, MakeOperators, MatchLiteral, Operator,
    };

    use super::{ExprIdxVec, DEPTH_PRIO_STEP};

    pub type FlatNodeVec<T> = SmallVec<[FlatNode<T>; N_NODES_ON_STACK]>;
    pub type FlatOpVec<T> = SmallVec<[FlatOp<T>; N_NODES_ON_STACK]>;
    type UnaryOpIdxDepthStack = SmallVec<[(usize, i64); N_UNARYOPS_OF_DEEPEX_ON_STACK]>;

    /// A `FlatOp` contains besides a binary operation an optional unary operation that
    /// will be executed after the binary operation in case of its existence.
    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
    pub struct FlatOp<T: Clone> {
        pub unary_op: UnaryOp<T>,
        pub bin_op: BinOp<T>,
    }

    impl<T: Clone> OperateBinary<T> for FlatOp<T> {
        fn apply(&self, arg1: T, arg2: T) -> T {
            self.unary_op.apply((self.bin_op.apply)(arg1, arg2))
        }
    }

    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
    pub enum FlatNodeKind<T> {
        Num(T),
        Var(usize),
    }

    #[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
    pub struct FlatNode<T> {
        pub kind: FlatNodeKind<T>,
        pub unary_op: UnaryOp<T>,
    }

    impl<T> FlatNode<T>
    where
        T: Clone,
    {
        pub(super) fn from_kind(kind: FlatNodeKind<T>) -> FlatNode<T> {
            FlatNode {
                kind,
                unary_op: UnaryOp::new(),
            }
        }
    }

    use crate::expression::deep::{BinOpsWithReprs, DeepEx, DeepNode, UnaryOpWithReprs};

    fn collect_reprs<'a, F, T, I>(
        funcs: I,
        ops: &[Operator<'a, T>],
        predicate: fn(&Operator<T>, F) -> bool,
    ) -> ExResult<SmallVec<[Operator<'a, T>; N_UNARYOPS_OF_DEEPEX_ON_STACK]>>
    where
        T: Clone,
        I: Iterator<Item = F>,
        F: Clone,
    {
        funcs
            .map(|func| {
                ops.iter()
                    .find(|op| predicate(op, func.clone()))
                    .cloned()
                    .ok_or_else(|| ExError::new("could not find operator"))
            })
            .collect::<ExResult<SmallVec<[Operator<'a, T>; N_UNARYOPS_OF_DEEPEX_ON_STACK]>>>()
    }

    fn unary_predicate<T: Clone>(op: &Operator<T>, func: &fn(T) -> T) -> bool {
        if op.has_unary() {
            op.unary().unwrap() == *func
        } else {
            false
        }
    }

    fn binary_predicate<T: Clone>(op: &Operator<T>, func: &fn(T, T) -> T) -> bool {
        if op.has_bin() {
            op.bin().unwrap().apply == *func
        } else {
            false
        }
    }

    fn collect_unary_reprs<'a, T: Clone>(
        ops: &[Operator<'a, T>],
        unary_op: &UnaryOp<T>,
    ) -> ExResult<SmallVec<[&'a str; N_UNARYOPS_OF_DEEPEX_ON_STACK]>> {
        Ok(collect_reprs::<&fn(T) -> T, _, _>(
            unary_op.funcs_to_be_composed().iter(),
            ops,
            unary_predicate,
        )?
        .iter()
        .map(|op| op.repr())
        .collect::<SmallVec<[&'a str; N_UNARYOPS_OF_DEEPEX_ON_STACK]>>())
    }

    fn convert_node<'a, T, OF, LM>(
        node: FlatNode<T>,
        var_names: &[String],
        ops: &[Operator<'a, T>],
    ) -> DeepNode<'a, T, OF, LM>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        let deepnode = match node.kind {
            FlatNodeKind::Num(n) => DeepNode::Num(n),
            FlatNodeKind::Var(var_idx) => DeepNode::Var((var_idx, var_names[var_idx].clone())),
        };

        // cannot fail unless there is a bug
        let reprs = collect_unary_reprs(ops, &node.unary_op).unwrap();

        let n_reprs = reprs.len();
        let unary_op = UnaryOpWithReprs {
            reprs,
            op: node.unary_op.clone(),
        };
        if n_reprs > 0 {
            DeepNode::Expr(Box::new(
                DeepEx::new(vec![deepnode], BinOpsWithReprs::<T>::new(), unary_op).unwrap(),
            )) // cannot fail unless there is a bug
        } else {
            deepnode
        }
    }

    pub(super) fn flatex_to_deepex<'a, T, OF, LM>(
        mut flat_ops: FlatOpVec<T>,
        nodes: FlatNodeVec<T>,
        var_names: SmallVec<[String; N_VARS_ON_STACK]>,
    ) -> ExResult<DeepEx<'a, T, OF, LM>>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
        <T as FromStr>::Err: Debug,
    {
        let dummy_node = DeepNode::Var((usize::MAX, "".to_string()));
        let operators = OF::make();
        let bin_ops = collect_reprs::<&fn(T, T) -> T, _, _>(
            flat_ops.iter().map(|op| &op.bin_op.apply),
            &operators,
            binary_predicate,
        )?;
        type BinVecT<T> = SmallVec<[T; N_UNARYOPS_OF_DEEPEX_ON_STACK]>;
        let bin_reprs = bin_ops
            .iter()
            .map(|op| op.repr())
            .collect::<BinVecT<&str>>();
        let orig_prios = bin_ops
            .iter()
            .map(|op| Ok(op.bin()?.prio))
            .collect::<ExResult<BinVecT<i64>>>()?;
        let prio_inds = prioritized_indices_flat(&flat_ops, &nodes);
        let mut deep_nodes = nodes
            .into_iter()
            .map(|dn| convert_node::<T, OF, LM>(dn, &var_names, &operators))
            .collect::<Vec<DeepNode<T, OF, LM>>>();
        let mut tracker: SmallVec<[usize; N_NODES_ON_STACK]> =
            smallvec![0; 1 + deep_nodes.len() / usize::BITS as usize];
        debug_assert!(deep_nodes.len() <= tracker.max_len());
        for &idx in &prio_inds {
            let shift_left = tracker.get_previous(idx);
            let shift_right = tracker.consume_next(idx);

            let num_1_idx = idx - shift_left;
            let num_2_idx = idx + shift_right;

            // point of panic for invalid input
            assert!(
                num_1_idx < deep_nodes.len()
                    && num_2_idx < deep_nodes.len()
                    && idx < flat_ops.len()
            );

            let bin_op = flat_ops[idx].bin_op.clone();
            let bin_op = BinOp {
                apply: bin_op.apply,
                prio: orig_prios[idx],
                is_commutative: bin_op.is_commutative,
            };
            let bin_op_wr = BinOpsWithReprs {
                reprs: smallvec![bin_reprs[idx]],
                ops: smallvec![bin_op],
            };
            let unary_op = mem::take(&mut flat_ops[idx].unary_op);
            let unary_reprs = collect_unary_reprs(&operators, &unary_op)?;
            let unary_op = UnaryOpWithReprs {
                reprs: unary_reprs,
                op: unary_op,
            };

            let deepex = DeepEx::new(
                vec![
                    mem::replace(&mut deep_nodes[num_1_idx], dummy_node.clone()),
                    mem::replace(&mut deep_nodes[num_2_idx], dummy_node.clone()),
                ],
                bin_op_wr,
                unary_op,
            )?;
            deep_nodes[num_1_idx] = DeepNode::Expr(Box::new(deepex));
        }
        let final_node = deep_nodes
            .first()
            .ok_or_else(|| format_exerr!("prio indices cannot be empty but is {:?}", prio_inds))?
            .clone();
        let mut deepex = DeepEx::new(
            vec![final_node],
            BinOpsWithReprs::new(),
            UnaryOpWithReprs::new(),
        )?;
        deepex.reset_vars(var_names.clone());
        deepex.compile();
        Ok(deepex)
    }

    fn eval_numbers<T: Clone + Debug + Default>(
        numbers: &mut SmallVec<[T; N_NODES_ON_STACK]>,
        ops: &[FlatOp<T>],
        prio_indices: &[usize],
    ) -> ExResult<T> {
        Ok(if numbers.len() <= usize::max_len(&0) {
            let mut ignore = 0;
            eval_binary(numbers.as_mut_slice(), ops, prio_indices, &mut ignore)
        } else {
            let mut ignore: SmallVec<[usize; N_NODES_ON_STACK]> =
                smallvec![0; 1 + numbers.len() / usize::BITS as usize];
            eval_binary(numbers.as_mut_slice(), ops, prio_indices, &mut ignore[..])
        })
    }

    pub(super) fn eval_flatex_cloning<T: Clone + Debug + Default>(
        vars: &[T],
        nodes: &[FlatNode<T>],
        ops: &[FlatOp<T>],
        prio_indices: &[usize],
    ) -> ExResult<T> {
        let mut numbers = nodes
            .iter()
            .map(|node| {
                node.unary_op.apply(match &node.kind {
                    FlatNodeKind::Num(n) => n.clone(),
                    FlatNodeKind::Var(idx) => vars[*idx].clone(),
                })
            })
            .collect::<SmallVec<[T; N_NODES_ON_STACK]>>();
        eval_numbers(&mut numbers, ops, prio_indices)
    }

    pub(super) fn eval_flatex_consuming_vars<T: Clone + Debug + Default>(
        vars: &mut [T],
        nodes: &[FlatNode<T>],
        ops: &[FlatOp<T>],
        prio_indices: &[usize],
    ) -> ExResult<T> {
        let mut numbers = nodes
            .iter()
            .map(|node| {
                node.unary_op.apply(match &node.kind {
                    FlatNodeKind::Num(n) => n.clone(),
                    FlatNodeKind::Var(idx) => mem::take(&mut vars[*idx]),
                })
            })
            .collect::<SmallVec<[T; N_NODES_ON_STACK]>>();

            eval_numbers(&mut numbers, ops, prio_indices)
    }

    /// This is called in case a closing paren occurs. If available, the index of the unary operator of the
    /// relevant depth operators will be returned and the open operator will be removed.
    ///   
    fn pop_unary_stack(unary_stack: &mut UnaryOpIdxDepthStack, depth: i64) -> Option<usize> {
        let last_idx_depth = unary_stack.last().copied();
        match last_idx_depth {
            Some((idx, d)) if d == depth => {
                unary_stack.pop();
                Some(idx)
            }
            _ => None,
        }
    }

    fn is_binary<'a, T>(
        op: &Operator<'a, T>,
        idx: usize,
        parsed_tokens: &[ParsedToken<'a, T>],
    ) -> ExResult<bool>
    where
        T: DataType,
    {
        Ok(idx > 0 && parser::is_operator_binary(op, &parsed_tokens[idx - 1])?)
    }

    type ExResultOption<T> = ExResult<Option<T>>;

    fn unpack_unary<T>(idx: usize, parsed_tokens: &[ParsedToken<T>]) -> ExResultOption<fn(T) -> T>
    where
        T: DataType,
    {
        match &parsed_tokens[idx] {
            ParsedToken::Op(op) => {
                if !is_binary(op, idx, parsed_tokens)? {
                    Ok(Some(op.unary()?))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    pub(super) fn make_expression<T, OF, LMF>(
        text: &str,
        parsed_tokens: &[ParsedToken<T>],
        parsed_vars: &[&str],
    ) -> ExResult<FlatEx<T, OF, LMF>>
    where
        T: DataType,
        OF: MakeOperators<T>,
        LMF: MatchLiteral,
    {
        let mut flat_nodes = FlatNodeVec::<T>::new();
        let mut flat_ops = FlatOpVec::<T>::new();

        let mut idx_tkn: usize = 0;
        let mut depth = 0;
        let mut unary_stack: UnaryOpIdxDepthStack = SmallVec::new();

        let iter_subsequent_unaries = |end_idx: usize| {
            let unpack = |idx| unpack_unary(idx, parsed_tokens);
            let dist_from_end = (0..end_idx + 1)
                .rev()
                .map(unpack)
                .take_while(|f| match f {
                    Ok(f) => f.is_some(),
                    _ => false,
                })
                .count();
            let start_idx = end_idx + 1 - dist_from_end;

            // check if we did terminate due to an error
            if start_idx > 0 {
                unpack(start_idx - 1)?;
            }

            Ok((start_idx..end_idx + 1).flat_map(unpack).flatten())
        };

        let create_node = |idx_node, kind| {
            if idx_node > 0 {
                let idx_op = idx_node - 1;
                if let ParsedToken::Op(op) = &parsed_tokens[idx_op] {
                    if !is_binary(op, idx_op, parsed_tokens)? {
                        return Ok(FlatNode {
                            kind,
                            unary_op: UnaryOp::from_iter(iter_subsequent_unaries(idx_op)?),
                        });
                    }
                }
            }
            Ok(FlatNode::from_kind(kind))
        };
        while idx_tkn < parsed_tokens.len() {
            match &parsed_tokens[idx_tkn] {
                ParsedToken::Op(op) => {
                    if is_binary(op, idx_tkn, parsed_tokens)? {
                        let mut bin_op = op.bin()?;
                        bin_op.prio += depth * DEPTH_PRIO_STEP;
                        flat_ops.push(FlatOp::<T> {
                            unary_op: UnaryOp::new(),
                            bin_op,
                        });
                    } else if let ParsedToken::Paren(p) = &parsed_tokens[idx_tkn + 1] {
                        match p {
                            Paren::Close => {
                                let err_msg =
                                    "a unary operator cannot on the left of a closing paren";
                                return Err(ExError::new(err_msg));
                            }
                            Paren::Open => unary_stack.push((idx_tkn, depth)),
                        };
                    }
                    idx_tkn += 1;
                }
                ParsedToken::Num(n) => {
                    let kind = FlatNodeKind::Num(n.clone());
                    let flat_node = create_node(idx_tkn, kind)?;
                    flat_nodes.push(flat_node);
                    idx_tkn += 1;
                }
                ParsedToken::Var(name) => {
                    let idx = parser::find_var_index(name, parsed_vars);
                    let kind = FlatNodeKind::Var(idx);
                    let flat_node = create_node(idx_tkn, kind)?;
                    flat_nodes.push(flat_node);
                    idx_tkn += 1;
                }
                ParsedToken::Paren(p) => {
                    match p {
                        Paren::Open => {
                            idx_tkn += 1;
                            depth += 1;
                        }
                        Paren::Close => {
                            let lowest_prio_flat_op = flat_ops
                                .iter_mut()
                                .rev()
                                .take_while(|op| op.bin_op.prio >= depth * DEPTH_PRIO_STEP)
                                .min_by(|fo1, fo2| fo1.bin_op.prio.cmp(&fo2.bin_op.prio));
                            match lowest_prio_flat_op {
                                None => {
                                    // no binary operators of current depth, attach to last node
                                    let last_node =
                                        flat_nodes.iter_mut().last().ok_or_else(|| {
                                            ExError::new("there must be a node between parens")
                                        })?;
                                    let mut closed = pop_unary_stack(&mut unary_stack, depth - 1);
                                    match &mut closed {
                                        None => (),
                                        Some(uop_idx) => last_node
                                            .unary_op
                                            .append_after_iter(iter_subsequent_unaries(*uop_idx)?),
                                    }
                                }
                                Some(lowpfo) => {
                                    let mut closed = pop_unary_stack(&mut unary_stack, depth - 1);
                                    match &mut closed {
                                        None => (),
                                        Some(uop_idx) => lowpfo
                                            .unary_op
                                            .append_after_iter(iter_subsequent_unaries(*uop_idx)?),
                                    }
                                }
                            }
                            idx_tkn += 1;
                            depth -= 1;
                        }
                    }
                }
            }
        }
        let indices = prioritized_indices_flat(&flat_ops, &flat_nodes);
        Ok(FlatEx {
            nodes: flat_nodes,
            flat_ops,
            prio_indices: indices,
            var_names: parsed_vars.iter().map(|s| s.to_string()).collect(),
            text: text.to_string(),
            dummy_ops_factory: PhantomData,
            dummy_literal_matcher_factory: PhantomData,
        })
    }

    pub(super) fn parse<T, OF, LMF>(text: &str, ops: &[Operator<T>]) -> ExResult<FlatEx<T, OF, LMF>>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
        OF: MakeOperators<T>,
        LMF: MatchLiteral,
    {
        let mut expr = parse_wo_compile(text, ops)?;
        expr.compile();
        Ok(expr)
    }

    pub fn parse_wo_compile<T, OF, LMF>(
        text: &str,
        ops: &[Operator<T>],
    ) -> ExResult<FlatEx<T, OF, LMF>>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
        OF: MakeOperators<T>,
        LMF: MatchLiteral,
    {
        let parsed_tokens = parser::tokenize_and_analyze(text, ops, LMF::is_literal)?;
        parser::check_parsed_token_preconditions(&parsed_tokens)?;
        let parsed_vars = parser::find_parsed_vars(&parsed_tokens);
        make_expression(text, &parsed_tokens[0..], &parsed_vars)
    }

    pub(super) fn prioritized_indices_flat<T: Clone + Debug>(
        ops: &[FlatOp<T>],
        nodes: &[FlatNode<T>],
    ) -> ExprIdxVec {
        let prio_increase =
            |bin_op_idx: usize| match (&nodes[bin_op_idx].kind, &nodes[bin_op_idx + 1].kind) {
                (FlatNodeKind::Num(_), FlatNodeKind::Num(_))
                    if ops[bin_op_idx].bin_op.is_commutative =>
                {
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
}
/// Flattened expressions make efficient evaluation possible.
/// Simplified, a flat expression consists of a [`SmallVec`](https://docs.rs/smallvec/)
/// of nodes and a [`SmallVec`](https://docs.rs/smallvec/) of operators that are applied
/// to the nodes in an order following operator priorities.
///
/// Creation of expressions is possible with the function [`parse`](crate::parse) which is equivalent to
/// [`FlatEx::parse`](FlatEx::parse).
///
/// ```rust
/// # use std::error::Error;
/// # fn main() -> Result<(), Box<dyn Error>> {
/// #
/// use exmex::prelude::*;
///
/// // create an expression by parsing a string
/// let expr = FlatEx::<f32>::parse("sin(1+y)*x")?;
/// assert!((expr.eval(&[1.5, 2.0])? - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The argument `&[1.5, 2.0]` in the call of [`eval`](FlatEx::eval) specifies the
/// variable values in the alphabetical order of the variable names.
/// In this example, we want to evaluate the expression for the varibale values `x=2.0` and `y=1.5`.
///
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct FlatEx<T, OF = FloatOpsFactory<T>, LM = NumberMatcher>
where
    T: Debug + Clone,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    nodes: FlatNodeVec<T>,
    flat_ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    var_names: SmallVec<[String; N_VARS_ON_STACK]>,
    text: String,
    dummy_ops_factory: PhantomData<OF>,
    dummy_literal_matcher_factory: PhantomData<LM>,
}

impl<T, OF, LMF> FlatEx<T, OF, LMF>
where
    T: DataType,
    OF: MakeOperators<T>,
    LMF: MatchLiteral,
{
    pub fn new(
        nodes: FlatNodeVec<T>,
        ops: FlatOpVec<T>,
        prio_indices: ExprIdxVec,
        var_names: SmallVec<[String; N_VARS_ON_STACK]>,
        text: String,
    ) -> Self {
        Self {
            nodes,
            flat_ops: ops,
            prio_indices,
            var_names,
            text,
            dummy_ops_factory: PhantomData,
            dummy_literal_matcher_factory: PhantomData,
        }
    }

    /// Executes calculations that can trivially be executed, e.g., multiplies two numbers that
    /// need to be multiplied anyway.
    pub fn compile(&mut self) {
        let mut num_inds = self.prio_indices.clone();
        let mut used_prio_indices = ExprIdxVec::new();

        let mut already_declined: SmallVec<[bool; N_NODES_ON_STACK]> =
            smallvec::smallvec![false; self.nodes.len()];

        for node in &mut self.nodes {
            if let FlatNodeKind::Num(num) = &node.kind {
                *node = FlatNode::from_kind(FlatNodeKind::Num(node.unary_op.apply(num.clone())));
            }
        }
        for (i, &bin_op_idx) in self.prio_indices.iter().enumerate() {
            let num_idx = num_inds[i];
            let node_1 = &self.nodes[num_idx];
            let node_2 = &self.nodes[num_idx + 1];
            if let (FlatNodeKind::Num(num_1), FlatNodeKind::Num(num_2)) =
                (node_1.kind.clone(), node_2.kind.clone())
            {
                if !(already_declined[num_idx] || already_declined[num_idx + 1]) {
                    let op_result =
                        self.flat_ops[bin_op_idx]
                            .unary_op
                            .apply((self.flat_ops[bin_op_idx].bin_op.apply)(num_1, num_2));
                    self.nodes[num_idx] = FlatNode::from_kind(FlatNodeKind::Num(op_result));
                    self.nodes.remove(num_idx + 1);
                    already_declined.remove(num_idx + 1);
                    // reduce indices after removed position
                    for num_idx_after in num_inds.iter_mut() {
                        if *num_idx_after > num_idx {
                            *num_idx_after -= 1;
                        }
                    }
                    used_prio_indices.push(bin_op_idx);
                } else {
                    already_declined[num_idx] = true;
                    already_declined[num_idx + 1] = true;
                }
            } else {
                already_declined[num_idx] = true;
                already_declined[num_idx + 1] = true;
            }
        }

        self.flat_ops = self
            .flat_ops
            .iter()
            .enumerate()
            .filter(|(i, _)| !used_prio_indices.contains(i))
            .map(|(_, op)| op.clone())
            .collect();

        self.prio_indices = detail::prioritized_indices_flat(&self.flat_ops, &self.nodes);
    }

    /// Parses into an expression without compilation. Allow slightly faster direct evaluation of strings.
    pub fn parse_wo_compile(text: &str) -> ExResult<Self>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
    {
        let ops = OF::make();
        detail::parse_wo_compile(text, &ops)
    }

    /// Returns the indices of the variables in the order of their occurrence during the
    /// operations
    pub fn var_indices_ordered(&self) -> SmallVec<[usize; N_VARS_ON_STACK]> {
        self.prio_indices
            .iter()
            .flat_map(|idx| {
                iter::once(match &self.nodes[*idx].kind {
                    FlatNodeKind::Num(_) => None,
                    FlatNodeKind::Var(var_idx) => Some(*var_idx),
                })
                .chain(iter::once(if *idx == self.prio_indices.len() - 1 {
                    match &self.nodes[*idx + 1].kind {
                        FlatNodeKind::Num(_) => None,
                        FlatNodeKind::Var(var_idx) => Some(*var_idx),
                    }
                } else {
                    None
                }))
            })
            .flatten()
            .collect::<SmallVec<[usize; N_VARS_ON_STACK]>>()
    }
    
    pub fn eval_iter(&self, vars: impl Iterator<Item=T>) -> ExResult<T> {
        let mut vars = vars.collect::<SmallVec<[T; N_VARS_ON_STACK]>>();
        if self.var_names.len() != vars.len() {
            return Err(format_exerr!(
                "expression contains {} vars which is different to the length {} of the passed slice",
                self.var_names.len(),
                vars.len()
            ));
        }
        detail::eval_flatex_consuming_vars(&mut vars, &self.nodes, &self.flat_ops, &self.prio_indices)
    }
}

impl<'a, T, OF, LM> Express<'a, T> for FlatEx<T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    type LiteralMatcher = LM;
    type OperatorFactory = OF;

    fn eval(&self, vars: &[T]) -> ExResult<T> {
        if self.var_names.len() != vars.len() {
            return Err(format_exerr!(
                "expression contains {} vars which is different to the length {} of the passed slice",
                self.var_names.len(),
                vars.len()
            ));
        }
        detail::eval_flatex_cloning(vars, &self.nodes, &self.flat_ops, &self.prio_indices)
    }

    fn eval_relaxed(&self, vars: &[T]) -> ExResult<T> {
        if self.var_names.len() > vars.len() {
            return Err(format_exerr!(
                "expression contains {} vars which is higher than the length {} of the passed slice",
                self.var_names.len(),
                vars.len()
            ));
        }
        detail::eval_flatex_cloning(vars, &self.nodes, &self.flat_ops, &self.prio_indices)
    }

    fn unparse(&self) -> &str {
        self.text.as_str()
    }
    fn var_names(&self) -> &[String] {
        &self.var_names
    }

    fn to_deepex(self) -> ExResult<DeepEx<'a, T, OF, LM>>
    where
        Self: Sized,
        T: DataType,
        <T as FromStr>::Err: Debug,
    {
        detail::flatex_to_deepex(self.flat_ops, self.nodes, self.var_names)
    }
    fn from_deepex(deepex: DeepEx<T, OF, LM>) -> ExResult<Self>
    where
        Self: Sized,
        T: DataType,
        <T as FromStr>::Err: Debug,
    {
        {
            let (nodes, ops) = flatten_vecs(&deepex, 0);
            let indices = detail::prioritized_indices_flat(&ops, &nodes);
            Ok(FlatEx::new(
                nodes,
                ops,
                indices,
                deepex
                    .var_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<SmallVec<_>>(),
                deepex.unparse().to_string(),
            ))
        }
    }
    fn parse(text: &'a str) -> ExResult<Self>
    where
        Self: Sized,
    {
        let ops = OF::make();
        detail::parse(text, &ops)
    }
}

/// The expression is displayed as a string created by [`unparse`](FlatEx::unparse).
impl<T, OF, LMF> Display for FlatEx<T, OF, LMF>
where
    T: DataType,
    OF: MakeOperators<T>,
    LMF: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unparsed = self.unparse();
        write!(f, "{unparsed}")
    }
}

#[cfg(feature = "partial")]
use crate::expression::partial::Differentiate;

pub fn flatten_vecs<T, OF, LM>(
    deep_expr: &DeepEx<T, OF, LM>,
    prio_offset: i64,
) -> (FlatNodeVec<T>, FlatOpVec<T>)
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
    <T as FromStr>::Err: Debug,
{
    use self::detail::FlatOp;

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
                .append_after(deep_expr.unary_op().op.clone());
        } else {
            flat_nodes[0]
                .unary_op
                .append_after(deep_expr.unary_op().op.clone());
        }
    }
    (flat_nodes, flat_ops)
}

impl<'a, T, OF, LM> Calculate<'a, T> for FlatEx<T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T> + Debug,
    LM: MatchLiteral + Debug,
    <T as FromStr>::Err: Debug,
{
}

#[cfg(feature = "partial")]
impl<'a, T, OF, LM> Differentiate<'a, T> for FlatEx<T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T> + Debug,
    LM: MatchLiteral + Debug,
    <T as FromStr>::Err: Debug,
{
}

#[cfg(test)]
use crate::util::assert_float_eq_f64;

#[test]
fn test_to_deepex() -> ExResult<()> {
    fn test(sut: &str, vars: &[f64]) -> ExResult<()> {
        println!(" --- sut - {}", sut);
        let fex = FlatEx::<f64>::parse(sut)?;
        let dex = fex.clone().to_deepex()?;
        println!("{:#?}", dex);
        assert_float_eq_f64(fex.eval(vars)?, dex.eval(vars)?);
        Ok(())
    }
    test("{x}+2.0*{y}", &[1.0, 0.5])?;
    test("({x}+2.0)*{y}", &[1.0, 0.5])?;
    test("({x}+2.0)*(2^{y})", &[1.0, 0.5])?;
    test("(1+{x}+2.0)*(2-2^{y})", &[1.0, 0.5])?;
    test("(1+{x}+2.0)*2", &[1.0])?;
    test("{x}+(2.0*{y})", &[1.0, 0.5])?;
    test("sin({y})", &[1.0])?;
    test("sin({y}) + sin({x})", &[2.0, 1.0])?;
    test("sin(1+{y})", &[1.0])?;
    test("sin(cos(1+{y}))", &[1.0])?;
    test("sin((1+{y})*z)", &[1.0, 2.0])?;
    test("cos(sin(1+{y})*z)", &[1.0, 2.0])?;
    test("{x}+sin(2.0*{y})", &[1.0, 2.0])?;
    test("z+sin(x)+cos(y)", &[1.0, 2.0, 3.0])?;
    test("sin(cos(sin(z)))", &[2.53])?;
    test("1/(x/y)*(2*x)", &[1.3, 0.5])?;
    test("+-+x", &[12341.234])?;
    test("-y*(x*(-(1-y))) + 1.7", &[1.2, 1.0])?;

    Ok(())
}

#[test]
fn test_flat_compile() -> ExResult<()> {
    fn test(text: &str, vars: &[f64], ref_val: f64, ref_len: usize) -> ExResult<()> {
        println!("testing {}...", text);
        let flatex = FlatEx::<f64>::parse(text)?;
        assert_float_eq_f64(flatex.eval(vars)?, ref_val);
        assert_eq!(flatex.nodes.len(), ref_len);
        println!("...ok.");
        Ok(())
    }

    test("1*sin(2-0.1)", &[], 1.9f64.sin(), 1)?;
    test("x*(2*(2*(2*4*8)))", &[1.0], 32.0 * 8.0, 2)?;
    test("1*sin(2-0.1) + x", &[1.0], 1.0 + 1.9f64.sin(), 2)?;
    test("1.0 * 3 * 2 * x / 2 / 3", &[2.0], 2.0, 4)?;
    test(
        "x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+2+3+7*sin(y)-z/sin(3.0/2/(1-x*4*1*1*1*1))",
        &[2.21, 2.0, 3.0],
        45.37365538326699,
        13,
    )?;
    test("x / 2 / 3", &[1.0], 1.0 / 6.0, 3)?;
    test("x * 2 / 3", &[1.0], 2.0 / 3.0, 2)?;
    test(
        "(({x}^2.0)*(({x}^1.0)*2.0))+((({x}^1.0)*2.0)*({x}^2.0))",
        &[2.21],
        43.175444,
        10,
    )?;
    test("(((a+x^2*x^2)))", &[3.0, 2.21], 26.854432810000002, 5)?;

    let flatex = FlatEx::<f64>::parse("1*sin(2-0.1) + x")?;
    match flatex.nodes[0].kind {
        FlatNodeKind::Num(n) => assert_float_eq_f64(n, 1.9f64.sin()),
        _ => unreachable!(),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => unreachable!(),
    }

    let flatex = FlatEx::<f64>::parse("y + 1 - cos(1/(1*sin(2-0.1))-2) + 2 + x")?;
    assert_eq!(flatex.nodes.len(), 3);
    match flatex.nodes[0].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 1),
        _ => unreachable!(),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Num(_) => (),
        _ => unreachable!(),
    }
    match flatex.nodes[2].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
        _ => unreachable!(),
    }
    Ok(())
}
