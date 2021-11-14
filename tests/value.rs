#[cfg(feature = "value")]
use exmex::{ExResult, Express, Val};
#[cfg(feature = "value")]
mod utils;
#[test]
#[cfg(feature = "value")]
fn test_vars() -> ExResult<()> {
    let expr = exmex::parse_val::<i32, f64>("x+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[Val::Float(3.4)])?.to_float()?, 8.7);

    let expr = exmex::parse_val_owned::<i32, f64>("-(x1 if x0 else x2)+5.3")?;
    let res = expr
        .eval(&[Val::Bool(true), Val::Float(3.4), Val::Int(3)])?
        .to_float()?;

    utils::assert_float_eq_f64(res, 1.9);

    let expr = exmex::parse_val_owned::<i64, f32>("-sin(x)+5.3")?;
    utils::assert_float_eq_f32(
        expr.eval(&[Val::Float(2.2)])?.to_float()?,
        -2.2f32.sin() + 5.3,
    );

    Ok(())
}

#[test]
#[cfg(feature = "value")]
fn test_readme() -> ExResult<()> {
    let expr = exmex::parse_val::<i32, f64>("0 if b < c else 1.2")?;
    let res = expr.eval(&[Val::Float(34.0), Val::Int(21)])?.to_float()?;
    assert!((res - 1.2).abs() < 1e-12);
    Ok(())
}
