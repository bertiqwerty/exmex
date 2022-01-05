use std::{fmt, fmt::Debug, marker::PhantomData, str::FromStr};

use serde::{de, de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

use crate::data_type::DataType;

use crate::{prelude::*, MatchLiteral, MakeOperators};

fn serialize<T: Clone, S: Serializer, Ex: Express<T>>(
    serializer: S,
    expr: &Ex,
) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(
        expr.unparse(),
    )
}

impl<'de, T: DataType, OF: MakeOperators<T>, LMF: MatchLiteral> Serialize
    for FlatEx<T, OF, LMF>
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize(serializer, self)
    }
}

impl<'de, T: DataType + 'de, OF: MakeOperators<T>, LMF: MatchLiteral> Deserialize<'de>
    for FlatEx<T, OF, LMF>
where
    <T as std::str::FromStr>::Err: Debug,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FlatExVisitor {
            lifetime_dummy: PhantomData,
            of_dummy: PhantomData,
            literal_matcher_dummy: PhantomData,
        })
    }
}

#[derive(Debug)]
struct FlatExVisitor<'a, T, OF, LMF> {
    lifetime_dummy: PhantomData<&'a T>,
    of_dummy: PhantomData<OF>,
    literal_matcher_dummy: PhantomData<LMF>,
}

impl<'de, T: DataType, OF: MakeOperators<T>, LMF: MatchLiteral> Visitor<'de>
    for FlatExVisitor<'de, T, OF, LMF>
where
    <T as std::str::FromStr>::Err: Debug,
{
    type Value = FlatEx<T, OF, LMF>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a borrowed &str that can be parsed by `exmex` crate")
    }

    fn visit_borrowed_str<E>(self, unparsed: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let flatex = Self::Value::from_str(unparsed);
        flatex.map_err(|epe| E::custom(format!("Parse error - {}", epe.msg)))
    }
}


#[cfg(test)]
use {
    crate::operators::{BinOp, Operator},
    crate::partial::Differentiate,
    serde_test::Token,
};

#[test]
fn test_ser_de() {
    let test_inner = |exp, s| {
        serde_test::assert_ser_tokens(&exp, &[Token::Str(s)]);
        let serialized = serde_json::to_string(&exp).unwrap();
        let deserialized = serde_json::from_str::<FlatEx<f64>>(serialized.as_str()).unwrap();
        assert_eq!(s, format!("{}", deserialized));
    };

    let test = |s, s_1| {
        let flatex = FlatEx::<f64>::from_str(s).unwrap();
        test_inner(flatex.clone(), s);

        let dflatex_dy = flatex.clone().partial(1).unwrap();
        test_inner(dflatex_dy.clone(), s_1);
    };

    test("{x}+{y}*2.0", "2.0");
    test("{x}+sin(2.0*{y})", "2.0*cos(2.0*{y})");
    test("1.0/{x}+cos({y})*2.0", "2.0*-(sin({y}))");
    test("{y}*{x}*2.0", "2.0*{x}");
}

#[test]
fn test_ser_de_non_float() {
    fn test(to_be_parsed: &str, ref_val: i32) {
        #[derive(Clone)]
        struct IntegerOps;
        impl MakeOperators<i32> for IntegerOps {
            fn make<'a>() -> Vec<Operator<'a, i32>> {
                vec![
                    Operator::make_bin(
                        "%",
                        BinOp {
                            apply: |a: i32, b: i32| a % b,
                            prio: 1,
                            is_commutative: false,
                        },
                    ),
                    Operator::make_bin(
                        "/",
                        BinOp {
                            apply: |a: i32, b: i32| a / b,
                            prio: 1,
                            is_commutative: false,
                        },
                    ),
                ]
            }
        }
        let expr = FlatEx::<i32, IntegerOps>::from_str(to_be_parsed).unwrap();
        let serialized = serde_json::to_string(&expr).unwrap();
        let deserialized =
            serde_json::from_str::<FlatEx<i32, IntegerOps>>(serialized.as_str()).unwrap();

        assert_eq!(deserialized.eval(&[1]).unwrap(), ref_val);
    }
    test("19 % 5 / 2 / a", 2);
    test("4 % 2 / a", 0);
    test("4 / 2 / a", 2);
    test("4 / 2 / 2 / a", 1);
}
