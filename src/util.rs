#[cfg(test)]
fn assert_float_eq<T: num::Float + std::fmt::Display>(f1: T, f2: T, tol: T) {
    if (f1 - f2).abs() >= tol {
        println!("Floats not almost equal.\nf1: {}\nf2: {}\n", f1, f2);
        assert!(false);
    }
}
#[cfg(test)]
pub fn assert_float_eq_f32(f1: f32, f2: f32) {
    assert_float_eq(f1, f2, 1e-6);
}
#[cfg(test)]
pub fn assert_float_eq_f64(f1: f64, f2: f64) {
    assert_float_eq(f1, f2, 1e-12);
}

