use crate::expression::flat_details::{self, FlatNodeVec, FlatOpVec};
use crate::{
    expression::{
        deep::{DeepBuf, DeepEx, ExprIdxVec},
        partial_derivatives,
    },
    operators,
    parser::ExParseError,
};
use num::Float;
use std::fmt::{self, Debug, Display, Formatter};

/// Flattens a deep expression
/// The result does not contain any recursive structures and is faster to evaluate.
pub fn flatten<T: Copy + Debug>(deepex: DeepEx<T>) -> FlatEx<T> {
    let (nodes, ops) = flat_details::flatten_vecs(&deepex, 0);
    let indices = flat_details::prioritized_indices_flat(&ops, &nodes);
    let n_unique_vars = deepex.n_vars();
    FlatEx {
        nodes,
        ops,
        prio_indices: indices,
        n_unique_vars,
        deepex: Some(deepex),
    }
}

/// This is the core data type representing a flattened expression and the result of
/// parsing a string. We use flattened expressions to make efficient evaluation possible.
/// Simplified, a flat expression consists of a [`SmallVec`](https://docs.rs/smallvec/) 
/// of nodes and a [`SmallVec`](https://docs.rs/smallvec/) of operators that are applied
/// to the nodes in an order following operator priorities.
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
/// assert!((expr.eval(&[1.5, 2.0])? - (1.0 + 2.0 as f32).sin() * 1.5).abs() < 1e-6);
/// #
/// #     Ok(())
/// # }
/// ```
/// The second argument `&[1.5, 2.0]` in the call of [`eval`](FlatEx::eval) specifies the
/// variable values in the alphabetical order of the variable names.
/// In this example, we want to evaluate the expression for the varibale values `x=2.0` and `y=1.5`.
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
    ///            the n-th variable in alphabetical order.
    ///            Thereby, only the first occurrence of the variable in the string is relevant.
    ///            If an expression has been created by partial derivation, the variables always
    ///            coincide with those of the antiderivatives even in cases where variables are
    ///            irrelevant such as `(x)'=1`.
    ///
    /// # Errors
    ///
    /// If the number of variables in the parsed expression are different from the length of
    /// the variable slice, we return an [`ExParseError`](ExParseError).
    ///
    pub fn eval(&self, vars: &[T]) -> Result<T, ExParseError> {
        flat_details::eval_flatex(
            vars,
            &self.nodes,
            &self.ops,
            &self.prio_indices,
            self.n_unique_vars,
        )
    }

    /// This method computes a `FlatEx` instance that is a partial derivative of `self` with default operators
    /// as shown in the following example.
    ///
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::{parse_with_default_ops};
    ///
    /// let expr = parse_with_default_ops::<f64>("sin(1+y^2)*x")?;
    /// let dexpr_dx = expr.clone().partial(0)?;
    /// let dexpr_dy = expr.partial(1)?;
    ///
    /// assert!((dexpr_dx.eval(&[9e5, 2.0])? - (5.0 as f64).sin()).abs() < 1e-12);
    /// //                   |    
    /// //             This partial derivative dexpr_dx does depend on x. Still, it expects
    /// //             the same number of parameters as the corresponding
    /// //             antiderivative. Hence, you can pass any number for x.  
    ///
    /// assert!((dexpr_dy.eval(&[2.5, 2.0])? - 10.0 * (5.0 as f64).cos()).abs() < 1e-12);
    /// #
    /// #     Ok(())
    /// # }
    /// ```
    /// # Arguments
    ///
    /// * `var_idx` - variable with respect to which the partial derivative is computed
    ///
    /// # Errors
    ///
    /// * If `self` has been `clear_deepex`ed, we cannot compute the partial derivative and return an [`ExParseError`](ExParseError).
    /// * If you use none-default operators this might not work as expected. It could return an [`ExParseError`](ExParseError) if
    ///   an operator is not found or compute a wrong result if an operator is defined in an un-expected way.
    ///
    pub fn partial(self, var_idx: usize) -> Result<Self, ExParseError>
    where
        T: Float,
    {
        let ops = operators::make_default_operators();

        let d_i = partial_derivatives::partial_deepex(
            var_idx,
            self.deepex.ok_or(ExParseError {
                msg: "need deep expression for derivation, not possible after calling `clear`"
                    .to_string(),
            })?,
            &ops,
        )?;
        Ok(flatten(d_i))
    }

    /// Creates an expression string that corresponds to the `FlatEx` instance. This is
    /// not necessarily the input string. More precisely,
    /// * variables are put between curly braces,
    /// * spaces outside of curly brackets are ignored,
    /// * parentheses can be different from the input, and
    /// * expressions are compiled
    /// as shown in the following example.
    /// ```rust
    /// # use std::error::Error;
    /// # fn main() -> Result<(), Box<dyn Error>> {
    /// #
    /// use exmex::parse_with_default_ops;
    /// let flatex = parse_with_default_ops::<f64>("--sin ( z) +  {another var} + 1 + 2")?;
    /// assert_eq!(format!("{}", flatex), "-(-(sin({z})))+{another var}+3.0");
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
    /// to reduce memory consumption. The methods [`partial`](FlatEx::partial) and 
    /// [`unparse`](FlatEx::unparse) as well as the implementation of the 
    /// [`Display`](std::fmt::Display) trait will stop working after calling this function.
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

/// This is another representation of a flattened expression besides [`FlatEx`](FlatEx).
/// The interface is identical. The difference is that [`OwnedFlatEx`](OwnedFlatEx) can be used without
/// a lifetime parameter. All the data that [`FlatEx`](FlatEx) borrowed is kept in a 
/// buffer by [`OwnedFlatEx`](OwnedFlatEx). The drawback is that parsing takes longer, since
/// additional allocations are necessary. Evaluation time should be about the same for 
/// [`FlatEx`](FlatEx) and [`OwnedFlatEx`](OwnedFlatEx).
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct OwnedFlatEx<T: Copy + Debug> {
    deepex_buf: Option<DeepBuf<T>>,
    nodes: FlatNodeVec<T>,
    ops: FlatOpVec<T>,
    prio_indices: ExprIdxVec,
    n_unique_vars: usize,
}
impl<T: Copy + Debug> OwnedFlatEx<T> {
    /// see [`FlatEx::eval`](FlatEx::eval)
    pub fn eval(&self, vars: &[T]) -> Result<T, ExParseError> {
        flat_details::eval_flatex(
            vars,
            &self.nodes,
            &self.ops,
            &self.prio_indices,
            self.n_unique_vars,
        )
    }
    /// Creates an `OwnedFlatEx` instance from an instance of `FlatEx`.
    pub fn from_flatex<'a>(flatex: FlatEx<'a, T>) -> Self {
        Self {
            deepex_buf: flatex.deepex.map(|d| DeepBuf::from_deepex(&d)),
            nodes: flatex.nodes,
            ops: flatex.ops,
            prio_indices: flatex.prio_indices,
            n_unique_vars: flatex.n_unique_vars,
        }
    }
    /// See [`FlatEx::partial`](FlatEx::partial).
    pub fn partial(self, var_idx: usize) -> Result<Self, ExParseError>
    where
        T: Float,
    {
        let ops = operators::make_default_operators();
        let deep_buf = match self.deepex_buf {
            Some(d) => Ok(d),
            None => Err(ExParseError {
                msg: "need deep expression for derivation, not possible after calling `clear`"
                    .to_string(),
            }),
        }?;
        let deepex = deep_buf.to_deepex(&ops)?;
        let d_i = partial_derivatives::partial_deepex(var_idx, deepex, &ops)?;
        Ok(Self::from_flatex(flatten(d_i)))
    }
    /// See [`FlatEx::unparse`](FlatEx::unparse).
    pub fn unparse(&self) -> Result<String, ExParseError> {
        match &self.deepex_buf {
            Some(deepex) => Ok(deepex.unparsed.clone()),
            None => Err(ExParseError {
                msg: "unparse impossible, since deep expression optimized away".to_string(),
            }),
        }
    }
    /// Clears buffer with owned data, see also [`FlatEx::unparse`](FlatEx::unparse).
    pub fn clear_deepex(&mut self) {
        self.deepex_buf = None;
    }
}
/// The expression is displayed as a string created by [`unparse`](OwnedFlatEx::unparse).
impl<T: Copy + Debug> Display for OwnedFlatEx<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unparsed = self.unparse();
        match unparsed {
            Err(e) => write!(f, "{}", e.msg),
            Ok(s) => write!(f, "{}", s),
        }
    }
}

#[cfg(test)]
use crate::{
    expression::{deep::UnaryOpWithReprs, flat_details::FlatNodeKind},
    operators::{UnaryOp, VecOfUnaryFuncs},
    parse_with_default_ops,
    util::assert_float_eq_f64,
};

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
        FlatNodeKind::Var(idx) => assert_eq!(idx, 1),
        _ => assert!(false),
    }
    match flatex.nodes[1].kind {
        FlatNodeKind::Num(_) => (),
        _ => assert!(false),
    }
    match flatex.nodes[2].kind {
        FlatNodeKind::Var(idx) => assert_eq!(idx, 0),
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
    println!("{}", flatex);
    assert_eq!(format!("{}", flatex), "sin({var})/5.0");
    flatex.clear_deepex();
    assert_eq!(
        format!("{}", flatex),
        "unparse impossible, since deep expression optimized away"
    );

    let flatex = flatten(DeepEx::<f64>::from_str("sin(var)/5").unwrap());
    let mut owned_flatex = OwnedFlatEx::from_flatex(flatex);
    assert_eq!(format!("{}", owned_flatex), "sin({var})/5.0");
    owned_flatex.clear_deepex();
    assert_eq!(
        format!("{}", owned_flatex),
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
    let text_ref = "5.0+{x}";
    test(text, text_ref);
    let text = "sin(5+var)^(1/{y})+{var}";
    let text_ref = "sin(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref);
    let text = "-(5+var)^(1/{y})+{var}";
    let text_ref = "-(5.0+{var})^(1.0/{y})+{var}";
    test(text, text_ref);
    let text = "cos(sin(-(5+var)^(1/{y})))+{var}";
    let text_ref = "cos(sin(-(5.0+{var})^(1.0/{y})))+{var}";
    test(text, text_ref);
    let text = "cos(sin(-5+var^(1/{y})))-{var}";
    let text_ref = "cos(sin(-5.0+{var}^(1.0/{y})))-{var}";
    test(text, text_ref);
    let text = "cos(sin(-z+var*(1/{y})))+{var}";
    let text_ref = "cos(sin(-({z})+{var}*(1.0/{y})))+{var}";
    test(text, text_ref);
}
