use std::{fmt::Debug, str::FromStr};

use num::Float;
use smallvec::SmallVec;

use crate::{
    data_type::DataType,
    definitions::N_VARS_ON_STACK,
    expression::flat,
    parser,
    partial::{details, parse, DeepEx, Differentiate},
    ExResult, Express, FlatEx, MakeOperators, MatchLiteral, Operator,
};

impl<T, OF, LM> Differentiate<T> for FlatEx<T, OF, LM>
where
    T: DataType,
    OF: MakeOperators<T>,
    LM: MatchLiteral,
{
    fn to_deepex<'a>(&'a self, ops: &[Operator<'a, T>]) -> ExResult<DeepEx<'a, T>>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug,
        FlatEx<T, OF, LM>: Express<T>,
    {
        let var_names = self
            .var_names()
            .iter()
            .map(AsRef::as_ref)
            .collect::<SmallVec<[&str; N_VARS_ON_STACK]>>();

        let mut deepex = parse(self.unparse(), ops, parser::is_numeric_text)?;
        deepex.reset_vars(var_names);
        Ok(deepex)
    }

    fn from_deepex(deepex: DeepEx<T>, _: &[Operator<T>]) -> ExResult<Self>
    where
        Self: Sized,
        T: DataType + Float,
        <T as FromStr>::Err: Debug,
    {
        {
            let (nodes, ops) = details::flatten_vecs(&deepex, 0);
            let indices = flat::prioritized_indices_flat(&ops, &nodes);
            Ok(FlatEx::new(
                nodes,
                ops,
                indices,
                deepex
                    .var_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<SmallVec<_>>(),
                deepex.unparse(),
            ))
        }
    }
}
