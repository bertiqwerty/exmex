use exmex::{Express, ExResult, make_tuple};
mod utils;
#[test]
fn test_vars() -> ExResult<()> {
    let expr = exmex::parse_val::<i32, f64>("x.0+5.3")?;
    let tuple = make_tuple!(i32, f64, (3.4, Float), (3, Int), (true, Bool));
    utils::assert_float_eq_f64(expr.eval(&[tuple])?.to_float()?, 8.7);
    Ok(())
}