use std::{fmt, fmt::Debug, marker::PhantomData, str::FromStr};

use num::Float;
use serde::{de, de::Visitor, Deserialize, Deserializer, Serialize, Serializer};

use crate::{expression::deep::DeepEx, expression::flat, expression::flat::FlatEx};

impl<'de: 'a, 'a, T: Copy + Debug> Serialize for FlatEx<'a, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(
            self.unparse()
                .map_err(|e| {
                    serde::ser::Error::custom(format!("serialization failed - {}", e.msg))
                })?
                .as_str(),
        )
    }
}

impl<'de: 'a, 'a, T: Float + Debug + FromStr + 'a> Deserialize<'de> for FlatEx<'a, T>
where
    <T as std::str::FromStr>::Err: Debug,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(FlatExVisitor {
            lifetime_dummy: PhantomData,
        })
    }
}

#[derive(Debug)]
struct FlatExVisitor<'a, T> {
    lifetime_dummy: PhantomData<&'a T>,
}

impl<'de: 'a, 'a, T: Float + Debug + FromStr> Visitor<'de> for FlatExVisitor<'a, T>
where
    <T as std::str::FromStr>::Err: Debug,
{
    type Value = FlatEx<'a, T>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a borrowed &str that can be parsed by `exmex` crate")
    }

    fn visit_borrowed_str<E>(self, unparsed: &'de str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let deepex = DeepEx::from_str(unparsed);
        match deepex {
            Ok(d) => Ok(flat::flatten(d)),
            Err(epe) => Err(E::custom(format!("Parse error - {}", epe.msg))),
        }
    }
}

#[cfg(test)]
use serde_test::Token;

#[test]
fn test_ser_de() {
    let expr_str = "{x}+{y}*2.0";

    let deepex = DeepEx::<f64>::from_str(expr_str).unwrap();
    let flatex = flat::flatten(deepex);
    serde_test::assert_ser_tokens(&flatex, &[Token::Str("{x}+{y}*2.0")]);

    let serialized = serde_json::to_string(expr_str).unwrap();
    let deserialized = serde_json::from_str::<FlatEx<f64>>(serialized.as_str()).unwrap();
    assert_eq!(expr_str, format!("{}", deserialized));
}
