use exmex::{make_tuple, ExResult, Express};
mod utils;
#[test]
fn test_vars() -> ExResult<()> {
    let tuple = make_tuple!(i32, f64, (true, Bool), (3.4, Float), (3, Int));
    let expr = exmex::parse_val::<i32, f64>("x.1+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[tuple.clone()])?.to_float()?, 8.7);
    let expr = exmex::parse_val_owned::<i32, f64>("-ifelse(x)+5.3")?;
    utils::assert_float_eq_f64(expr.eval(&[tuple.clone()])?.to_float()?, 1.9);

    Ok(())
}
