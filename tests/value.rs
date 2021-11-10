#[cfg(feature = "value")]
use exmex::{make_tuple, ExResult, Express, Val};
#[cfg(feature = "value")]
mod utils;
#[test]
#[cfg(feature = "value")]
fn test_vars() -> ExResult<()> {
    let tuple = make_tuple!(i32, f64, (true, Bool), (3.4, Float), (3, Int));
    let expr = exmex::parse_val::<i32, f64>("x.1+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[tuple.clone()])?.to_float()?, 8.7);
    let expr = exmex::parse_val_owned::<i32, f64>("-(x.1 if x.0 else x.2)+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[tuple.clone()])?.to_float()?, 1.9);

    let expr = exmex::parse_val_owned::<i64, f32>("-sin(x)+5.3")?;
    utils::assert_float_eq_f32(
        expr.eval(&[Val::<i64, f32>::from_float(2.2)])?.to_float()?,
        -2.2f32.sin()+5.3,
    );

    Ok(())
}
