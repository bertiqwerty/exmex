use smallvec::SmallVec;

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

/// Container of unary operators of one expression
pub type UnaryOpVec<T> = SmallVec<[fn(T) -> T; 8]>;


/// Applies unary operators one after the other starting with the last.
/// # Arguments
///
/// * `uops` - unary operators to be applied
/// * `x` - number the unary operators are applied to
///
pub fn apply_unary_ops<T>(uops: &UnaryOpVec<T>, x: T) -> T {
    let mut result = x;
    // rev, since the last uop is applied first by convention
    for uo in uops.iter().rev() {
        result = uo(result);
    }
    result
}
