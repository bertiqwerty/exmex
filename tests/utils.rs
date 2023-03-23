pub fn assert_float_eq<T: num::Float + std::fmt::Display>(
    f1: T,
    f2: T,
    atol: T,
    rtol: T,
    msg: &str,
) {
    println!("tol {}", atol + rtol * f2.abs());
    println!("d   {}", (f1 - f2).abs());
    if (f1 - f2).abs() >= atol + rtol * f2.abs() {
        println!("Floats not almost equal. {}\nf1: {}\nf2: {}\n", msg, f1, f2);
        unreachable!();
    }
}

#[cfg(test)]
pub fn assert_float_eq_f64(f1: f64, f2: f64) {
    assert_float_eq(f1, f2, 1e-12, 0.0, "");
}
