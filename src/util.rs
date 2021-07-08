pub mod tests {
    pub fn assert_float_eq(f1: f32, f2: f32) {
        if (f1 - f2).abs() >= 1e-6 {
            panic!("Floats not almost equal.\nf1: {}\nf2: {}\n", f1, f2);
        }
    }
}

/// Applies unary operators one after the other starting with the last.
/// # Arguments
///
/// * `uops` - unary operators to be applied
/// * `x` - number the unary operators are applied to
///
pub fn apply_unary_ops<T>(uops: &Vec<fn(T) -> T>, x: T) -> T {
    let mut result = x;
    // rev, since the last uop is applied first by convention
    for uo in uops.iter().rev() {
        result = uo(result);
    }
    result
}
