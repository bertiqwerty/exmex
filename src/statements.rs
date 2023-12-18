#[cfg(feature = "value")]
use crate::{Val, ValMatcher, ValOpsFactory};
use std::fmt::Debug;
use std::str::FromStr;

use crate::{exerr, Calculate, DeepEx, FlatEx, MakeOperators, MatchLiteral};
use crate::{DataType, ExResult, Express};

fn eval_expr<T, OF, LM>(expr: FlatEx<T, OF, LM>, statements: &Statements<T, OF, LM>) -> ExResult<T>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    let mut deepex = expr.to_deepex()?;
    let mut f = |v: &str| {
        let idx = statements
            .vars
            .iter()
            .enumerate()
            .find(|(_, vs)| v == *vs)
            .map(|(i, _)| i);
        idx.and_then(|i| match &statements.expressions[i] {
            Rhs::Expr(expr) => expr.clone().to_deepex().ok(),
            Rhs::Val(v) => Some(DeepEx::from_num(v.clone())),
        })
    };
    deepex = deepex.subs(&mut f)?;
    deepex.eval(&[])
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum Rhs<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    Val(T),
    Expr(FlatEx<T, OF, LM>),
}
impl<T, OF, LM> Rhs<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    pub fn eval(self, statements: &Statements<T, OF, LM>) -> ExResult<T> {
        match self {
            Rhs::Val(v) => Ok(v.clone()),
            Rhs::Expr(expr) => eval_expr(expr, statements),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Statements<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    expressions: Vec<Rhs<T, OF, LM>>,
    vars: Vec<String>,
}

#[cfg(feature = "value")]
pub type StatementsVal<I, F> = Statements<Val<I, F>, ValOpsFactory, ValMatcher>;

impl<T, OF, LM> Statements<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    pub fn insert(mut self, var: &str, rhs: Rhs<T, OF, LM>) -> ExResult<Self> {
        let value = match rhs {
            Rhs::Expr(expr) => eval_expr(expr, &self)?,
            Rhs::Val(v) => v.clone(),
        };
        let exists_idx = self.vars.iter().position(|v| v == var);
        if let Some(idx) = exists_idx {
            self.vars[idx] = var.to_string();
            self.expressions[idx] = Rhs::Val(value);
        } else {
            self.vars.push(var.to_string());
            self.expressions.push(Rhs::Val(value));
        }
        Ok(self)
    }
}

pub struct Statement<'a, T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral
{
    pub var: Option<&'a str>,
    pub rhs: Rhs<T, OF, LM>
}

pub fn line_2_statement< T, OF, LM>(line_str: & str) -> ExResult<Statement< T, OF, LM>>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    let (var, expr) = parse(line_str)?;
    let rhs = if expr.var_names().is_empty() {
        Rhs::Val(expr.eval(&[])?)
    } else {
        Rhs::Expr(expr)
    };
    Ok(Statement{var, rhs})
}

type ParsedStatement<'a, T, OF, LM> = ExResult<(Option<&'a str>, FlatEx<T, OF, LM>)>;

fn parse<T, OF, LM>(s: &str) -> ParsedStatement<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    let mut splitted = s.split('=');
    let first = splitted.next();
    let second = splitted.next();
    Ok(match (first, second) {
        (Some(s), None) => (None, FlatEx::parse(s)?),
        (Some(var), Some(expr_str)) => (Some(var.trim()), FlatEx::parse(expr_str)?),
        _ => Err(exerr!("could not split {s}",))?,
    })
}

#[cfg(test)]
use crate::{FloatOpsFactory, NumberMatcher};
#[test]
fn test_statements() {
    let s = "x = 123";
    let Statement{var, rhs} = line_2_statement::<f32, FloatOpsFactory<f32>, NumberMatcher>(s).unwrap();
    assert_eq!(var, Some("x"));
    assert_eq!(rhs, Rhs::Val(123.0));
}
