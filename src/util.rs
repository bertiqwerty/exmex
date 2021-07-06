
pub fn apply_unary_ops<T>(uops: &Vec<fn(T)->T>, n: T) -> T {
    let mut result = n;
    // rev, since the last uop is applied first by convention
    for uo in uops.iter().rev() {
        result = uo(result);
    }
    result
}
