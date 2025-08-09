//! Work in progress

use std::fmt::Debug;
use std::str::FromStr;
#[cfg(feature = "value")]
use {
    crate::{Val, ValMatcher, ValOpsFactory},
    num::{Float, PrimInt, Signed},
};

use crate::{
    exerr, Calculate, DeepEx, FlatEx, FloatOpsFactory, MakeOperators, MatchLiteral, NumberMatcher,
};
use crate::{DataType, ExResult, Express};

use self::detail::{ParsedLhs, ParsedStatement};

mod detail {
    use std::{fmt::Debug, str::FromStr};

    use crate::{exerr, DataType, ExResult, Express, FlatEx, MakeOperators, MatchLiteral};

    #[allow(dead_code)]
    #[derive(Default, Debug)]
    pub enum ParsedLhs<'a> {
        Var(&'a str),
        Fn(Vec<&'a str>),
        #[default]
        None,
    }
    pub struct ParsedStatement<'a, T, OF, LM>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
    {
        pub expr: FlatEx<T, OF, LM>,
        pub lhs: ParsedLhs<'a>,
    }

    fn parse_func_str(fn_str: &str) -> Vec<&str> {
        fn_str
            .split(' ')
            .map(|s| s.trim().trim_matches(','))
            .collect()
    }

    pub fn parse<T, OF, LM>(s: &str) -> ExResult<ParsedStatement<'_, T, OF, LM>>
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
            (Some(s), None) => ParsedStatement {
                lhs: ParsedLhs::None,
                expr: FlatEx::parse(s)?,
            },
            (Some(s), Some(expr_str)) => {
                let s = s.trim();
                let expr = FlatEx::parse(expr_str)?;
                if s.contains(' ') || s.contains('(') {
                    ParsedStatement {
                        lhs: ParsedLhs::Fn(parse_func_str(s)),
                        expr,
                    }
                } else {
                    ParsedStatement {
                        lhs: ParsedLhs::Var(s),
                        expr,
                    }
                }
            }
            _ => Err(exerr!("could not split {s}",))?,
        })
    }
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
        println!("{self:?}");
        match self {
            Rhs::Val(v) => Ok(v.clone()),
            Rhs::Expr(expr) => statements.eval(expr),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Statements<T, OF = FloatOpsFactory<T>, LM = NumberMatcher>
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
pub type StatementsVal<I, F> = Statements<Val<I, F>, ValOpsFactory<I, F>, ValMatcher>;

impl<T, OF, LM> Statements<T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    pub fn insert(mut self, var: &str, rhs: Rhs<T, OF, LM>) -> Self {
        let exists_idx = self.vars.iter().position(|v| v == var);
        if let Some(idx) = exists_idx {
            self.vars[idx] = var.to_string();
            self.expressions[idx] = rhs;
        } else {
            self.vars.push(var.to_string());
            self.expressions.push(rhs);
        }
        self
    }
    fn eval(&self, expr: FlatEx<T, OF, LM>) -> ExResult<T>
    where
        T: DataType,
        <T as FromStr>::Err: Debug,
        OF: MakeOperators<T>,
        LM: MatchLiteral,
    {
        let mut deepex = expr.to_deepex()?;
        let mut f = |v: &str| {
            let idx = self
                .vars
                .iter()
                .enumerate()
                .find(|(_, vs)| v == *vs)
                .map(|(i, _)| i);
            idx.and_then(|i| match &self.expressions[i] {
                Rhs::Expr(expr) => expr.clone().to_deepex().ok(),
                Rhs::Val(v) => Some(DeepEx::from_num(v.clone())),
            })
        };
        deepex = deepex.subs(&mut f)?;
        deepex.eval(&[])
    }
}

pub struct Statement<'a, T, OF, LM>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    pub var: Option<&'a str>,
    pub rhs: Rhs<T, OF, LM>,
}

#[cfg(feature = "value")]
type StatementVal<'a, I, F> = Statement<'a, Val<I, F>, ValOpsFactory<I, F>, ValMatcher>;
#[cfg(feature = "value")]
pub fn line_2_statement_val<I, F>(line_str: &str) -> ExResult<StatementVal<'_, I, F>>
where
    I: DataType + Signed + PrimInt,
    F: DataType + Float,
    <I as FromStr>::Err: Debug,
    <F as FromStr>::Err: Debug,
{
    line_2_statement::<Val<I, F>, ValOpsFactory<I, F>, ValMatcher>(line_str)
}

pub fn line_2_statement<T, OF, LM>(line_str: &str) -> ExResult<Statement<'_, T, OF, LM>>
where
    T: DataType,
    <T as FromStr>::Err: Debug,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    let ParsedStatement { lhs: var, expr } = detail::parse(line_str)?;
    let rhs = if expr.var_names().is_empty() {
        Rhs::Val(expr.eval(&[])?)
    } else {
        Rhs::Expr(expr)
    };
    match var {
        ParsedLhs::Var(var) => Ok(Statement {
            var: Some(var),
            rhs,
        }),
        ParsedLhs::None => Ok(Statement { var: None, rhs }),
        _ => Err(exerr!("unsuported {:?}", var)),
    }
}

#[cfg(feature = "value")]
#[test]
fn test_statements() {
    let s = "x = 123";
    let Statement { var, rhs } = line_2_statement_val(s).unwrap();
    assert_eq!(var, Some("x"));
    assert_eq!(rhs, Rhs::Val(Val::Float(123.0)));
    let s = StatementsVal::<i32, f64>::default();
    let s = s.insert(var.unwrap(), rhs);
    let Statement { var: _, rhs } = line_2_statement_val("x").unwrap();
    assert_eq!(rhs.eval(&s).unwrap(), Val::Float(123.0));
}
