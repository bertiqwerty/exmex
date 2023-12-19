#[cfg(feature = "value")]
use exmex::{exerr, ExResult, Express, FlatExVal, Val};

#[cfg(feature = "value")]
mod utils;
#[test]
#[cfg(feature = "value")]
fn test_vars() -> ExResult<()> {
    let expr = exmex::parse_val::<i32, f64>("x+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[Val::Float(3.4)])?.to_float()?, 8.7);

    let expr = exmex::parse_val::<i32, f64>("-(x1 if x0 else x2)+5.3")?;
    let res = expr
        .eval(&[Val::Bool(true), Val::Float(3.4), Val::Int(3)])?
        .to_float()?;

    utils::assert_float_eq_f64(res, 1.9);

    let expr = exmex::parse_val::<i64, f32>("-sin(x)+5.3")?;
    utils::assert_float_eq::<f32>(
        expr.eval(&[Val::Float(2.2)])?.to_float()?,
        -2.2f32.sin() + 5.3,
        1e-6,
        0.0,
        "",
    );

    let expr = exmex::parse_val::<i64, f32>("-sin(x) if y > 0 else z + 3")?;
    utils::assert_float_eq::<f32>(
        expr.eval(&[Val::Float(1.0), Val::Int(2), Val::Int(3)])?
            .to_float()?,
        -1f32.sin(),
        1e-6,
        0.0,
        "",
    );
    assert_eq!(
        expr.eval(&[Val::Float(1.0), Val::Int(-1), Val::Int(3)])?
            .to_int()?,
        6,
    );

    let expr = exmex::parse_val::<i32, f64>("z if false else 2")?;
    println!("{:#?}", expr);
    assert_eq!(expr.eval(&[Val::Int(-3)])?.to_int()?, 2,);

    Ok(())
}

#[test]
#[cfg(feature = "value")]
#[cfg(feature = "partial")]
fn test_value_partial() -> ExResult<()> {
    use exmex::Differentiate;

    use crate::utils::assert_float_eq_f64;

    let expr = exmex::parse_val::<i32, f64>("x if x > 0 else (2*x if x >= -1 else -x)")?;

    assert_float_eq_f64(1.0, expr.eval(&[Val::Float(1.0)])?.to_float()?);
    assert_float_eq_f64(0.0, expr.eval(&[Val::Float(0.0)])?.to_float()?);
    assert_float_eq_f64(2.0, expr.eval(&[Val::Float(-2.0)])?.to_float()?);
    assert_float_eq_f64(-2.0, expr.eval(&[Val::Float(-1.0)])?.to_float()?);

    println!("{expr}");
    let deri = expr.partial(0).unwrap();

    for x in [-2.0, -1.5] {
        let res = deri.eval(&[Val::Float(x)])?.to_float()?;
        assert_float_eq_f64(res, -1.0);
    }
    for x in [1.0, 0.5, 3.0] {
        let res = deri.eval(&[Val::Float(x)])?.to_float()?;
        assert_float_eq_f64(res, 1.0);
    }
    for x in [-1.0, -0.5, 0.0] {
        let res = deri.eval(&[Val::Float(x)])?.to_float()?;
        assert_float_eq_f64(res, 2.0);
    }
    let sin = exmex::parse_val::<i32, f64>("sin(x)")?;
    let cos = sin.partial(0).unwrap();
    let res = cos.eval(&[Val::Float(34.0)])?.to_float()?;
    assert_float_eq_f64(res, 34.0f64.cos());

    let sin = exmex::parse_val::<i32, f64>("sin(x) if x < 0 else sin(x)")?;
    let cos = sin.partial(0).unwrap();
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let res = cos.eval(&[Val::Float(x)])?.to_float()?;
        assert_float_eq_f64(res, x.cos());
    }

    Ok(())
}

#[test]
#[cfg(feature = "value")]
fn test_readme() -> ExResult<()> {
    let expr = exmex::parse_val::<i32, f64>("0 if b < c else 1.2")?;
    let res = expr.eval(&[Val::Float(34.0), Val::Int(21)])?.to_float()?;
    assert!((res - 1.2).abs() < 1e-12);

    #[cfg(feature = "partial")]
    {
        use exmex::Differentiate;
        let expr = exmex::parse_val::<i32, f64>("3*x if x > 1 else x^2")?;
        let deri = expr.partial(0)?;
        let res = deri.eval(&[Val::Float(1.0)])?.to_float()?;
        assert!((res - 2.0).abs() < 1e-12);
        let res = deri.eval(&[Val::Float(7.0)])?.to_float()?;
        assert!((res - 3.0).abs() < 1e-12);
    }

    Ok(())
}

#[test]
#[cfg(feature = "serde")]
#[cfg(feature = "value")]
fn test_serde_public() -> ExResult<()> {
    use exmex::FlatExVal;

    let s = "{x}^3.0 if z < 0 else y";

    // flatex
    let flatex = FlatExVal::<i32, f64>::parse(s)?;
    let serialized = serde_json::to_string(&flatex).unwrap();
    let deserialized = serde_json::from_str::<FlatExVal<i32, f64>>(serialized.as_str()).unwrap();
    assert_eq!(deserialized.var_names().len(), 3);
    let res = deserialized.eval(&[Val::Float(2.0), Val::Bool(false), Val::Float(1.0)])?;
    assert_eq!(res.to_bool()?, false);
    let res = deserialized.eval(&[Val::Float(2.0), Val::Float(1.0), Val::Int(-1)])?;
    utils::assert_float_eq_f64(res.to_float()?, 8.0);
    assert_eq!(s, format!("{}", deserialized));
    Ok(())
}
#[cfg(feature = "value")]
#[test]
fn test_to() -> ExResult<()> {
    utils::assert_float_eq_f64(
        Val::<i32, f64>::Float(std::f64::consts::TAU).to_float()?,
        std::f64::consts::TAU,
    );
    assert_eq!(Val::<i32, f64>::Int(123).to_int()?, 123);
    assert!(Val::<i32, f64>::Bool(true).to_bool()?);
    assert_eq!(Val::<i32, f64>::Bool(false).to_int()?, 0);
    assert_eq!(Val::<i32, f64>::Bool(true).to_float()?, 1.0);
    utils::assert_float_eq_f64(Val::<i32, f64>::Float(3.4).to_float()?, 3.4);
    assert_eq!(Val::<i32, f64>::Int(34).to_int()?, 34);
    assert!(!Val::<i32, f64>::Bool(false).to_bool()?);
    Ok(())
}
#[cfg(feature = "value")]
#[test]
fn test_no_vars() -> ExResult<()> {
    fn test_int(s: &str, reference: i32) -> ExResult<()> {
        println!("=== testing\n{}", s);
        let res = exmex::parse_val::<i32, f64>(s)?.eval(&[])?.to_int();
        match res {
            Ok(i) => {
                assert_eq!(reference, i);
            }
            Err(e) => {
                println!("{:?}", e);
                unreachable!();
            }
        }
        Ok(())
    }
    fn test_float(s: &str, reference: f64) -> ExResult<()> {
        println!("=== testing\n{}", s);
        let expr = FlatExVal::<i32, f64>::parse(s)?;
        utils::assert_float_eq_f64(reference, expr.eval(&[])?.to_float()?);
        Ok(())
    }
    fn test_bool(s: &str, reference: bool) -> ExResult<()> {
        println!("=== testing\n{}", s);
        let expr = FlatExVal::<i32, f64>::parse(s)?;
        assert_eq!(reference, expr.eval(&[])?.to_bool()?);
        Ok(())
    }
    fn test_error(s: &str) -> ExResult<()> {
        let expr = FlatExVal::<i32, f64>::parse(s);
        match expr {
            Ok(exp) => {
                let v = exp.eval(&[])?;
                match v {
                    Val::Error(e) => {
                        println!("found expected error {:?}", e);
                        Ok(())
                    }
                    _ => Err(exerr!("'{}' should fail but didn't", s)),
                }
            }
            Err(e) => {
                println!("found expected error {:?}", e);
                Ok(())
            }
        }
    }
    fn test_none(s: &str) -> ExResult<()> {
        let expr = FlatExVal::<i32, f64>::parse(s)?;
        match expr.eval(&[])? {
            Val::None => Ok(()),
            _ => Err(exerr!("'{}' should return none but didn't", s)),
        }
    }
    test_int("1+2 if 1 > 0 else 2+4", 3)?;
    test_int("1+2 if 1 < 0 else 2+4", 6)?;
    test_error("929<<92")?;
    test_error("929<<32")?;
    test_error("929>>32")?;
    test_int("928<<31", 0)?;
    test_int("929>>31", 0)?;
    test_float("2.0^2", 4.0)?;
    test_int("2^4", 16)?;
    test_error("2^-4")?;
    test_int("2+4", 6)?;
    test_int("9+4", 13)?;
    test_int("9+4^2", 25)?;
    test_float("τ/TAU", 1.0)?;
    test_int("9/4", 2)?;
    test_int("9%4", 1)?;
    test_float("2.5+4.0^2", 18.5)?;
    test_float("2.5*4.0^2", 2.5 * 4.0 * 4.0)?;
    test_float("2.5-4.0^-2", 2.5 - 4.0f64.powi(-2))?;
    test_float("9.0/4.0", 9.0 / 4.0)?;
    test_float("sin(9.0)", 9.0f64.sin())?;
    test_float("cos(91.0)", 91.0f64.cos())?;
    test_float("ln(91.0)", 91.0f64.ln())?;
    test_float("log(91.0)", 91.0f64.ln())?;
    test_float("tan(913.0)", 913.0f64.tan())?;
    test_float("sin(-π)", 0.0)?;
    test_float("sin(π)", 0.0)?;
    test_float("τ", std::f64::consts::PI * 2.0)?;
    test_float("sin(-τ)", 0.0)?;
    test_float("round(π)", 3.0)?;
    test_float("cos(π)", -1.0)?;
    test_float("cos(TAU)", 1.0)?;
    test_float("sin (1 if false else 2.0)", 2.0f64.sin())?;
    test_float("cbrt(27.0)", 3.0)?;
    test_int("1 if true else 2.0", 1)?;
    test_float("(9.0 if true else 2.0)", 9.0)?;
    test_int("1<<4-2", 4)?;
    test_int("4>>2", 1)?;
    test_int("signum(4>>1)", 1)?;
    test_float("signum(-123.12)", -1.0)?;
    test_float("abs(-123.12)", 123.12)?;
    test_int("fact(4)", 2 * 3 * 4)?;
    test_int("fact(0)", 1)?;
    test_error("fact(-1)")?;
    test_bool("1>2", false)?;
    test_bool("1<2", true)?;
    test_bool("1.4>=1.4", true)?;
    test_bool("true==true", true)?;
    test_bool("false==true", false)?;
    test_bool("1.5 != 1.5 + 2.0", true)?;
    test_float("1 + 1.0", 2.0)?;
    test_bool("1.0 == 1", true)?;
    test_bool("1 == 1", true)?;
    test_bool("2 == true", false)?;
    test_bool("1.5 < 1", false)?;
    test_bool("true == true", true)?;
    test_bool("false != true", true)?;
    test_bool("false != false", false)?;
    test_bool("1 > 0.5", true)?;
    test_error("to_float(10000000000000)")?;
    test_bool("true == 1", false)?;
    test_bool("true else 2", true)?;
    test_int("1 else 2", 1)?;
    test_error("if true else 2")?;
    test_none("2 if false")?;
    test_int("to_int(1)", 1)?;
    test_int("to_int(3.5)", 3)?;
    test_float("to_float(2)", 2.0)?;
    test_float("to_float(3.5)", 3.5)?;
    test_float("to_float(true)", 1.0)?;
    test_float("to_float(false)", 0.0)?;
    test_int("to_int(true)", 1)?;
    test_int("to_int(false)", 0)?;
    test_error("to_int(fact(-1))")?;
    test_error("to_float(5 if false)")?;
    test_error("0/0")?;
    test_bool("(5 if false) == (5 if false)", false)?;
    test_error("2^40")?;
    test_error("1000000000*1000000000")?;
    test_error("1500000000+1500000000")?;
    test_error("-1500000000-1500000000")?;
    test_error("0%0")?;
    test_int("1&&2", 1)?;
    test_bool("true&&false", false)?;
    test_bool("false || true", true)?;
    test_int("1&&2.0", 1)?;
    test_float("1||2.0", 2.0)?;

    Ok(())
}

#[cfg(feature = "value")]
#[cfg(feature = "serde")]
#[test]
fn test_serde() {
    use serde::{Deserialize, Serialize};
    let s = "-1200 if (cb / ib) < 1 else -2400";
    let expr = FlatExVal::<i32, f64>::parse(s).unwrap();

    #[derive(Serialize, Deserialize)]
    struct Tmp {
        expr: FlatExVal<i32, f64>,
    }
    let tmp = Tmp { expr };
    let ser = serde_json::to_string_pretty(&tmp)
        .unwrap()
        .replace("/", "\\/");
    let _deser: Tmp = serde_json::from_str(&ser).unwrap();
}

#[cfg(feature = "value")]
#[test]
fn test_fuzz() {
    let s = "ata---n-----0>>220>22--ata---n-----0>>220>22-------------tanh-------------------tanh--------6/π";
    let expr = FlatExVal::<i64, f64>::parse(s).unwrap();

    let res = expr.eval(&[Val::Int(2), Val::Int(3)]).unwrap();
    assert!(!res.to_bool().unwrap());
    let s = "fact+82";
    let expr = FlatExVal::<i64, f64>::parse(s).unwrap();
    assert!(expr.eval(&[]).unwrap().to_int().is_err());
    assert!(expr.eval(&[]).unwrap().to_float().is_err());
    assert!(expr.eval(&[]).unwrap().to_bool().is_err());
}
