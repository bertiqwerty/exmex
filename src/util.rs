pub fn assert_float_eq(f1: f32, f2: f32) {
    if (f1 - f2).abs() >= 1e-5 {
        panic!("Floats not almost equal.\nf1: {}\nf2: {}\n", f1, f2);
    }
}
