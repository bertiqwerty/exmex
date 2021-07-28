use smallvec::{SmallVec, smallvec};
use std::fmt::Debug;

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

pub type VecOfUnaryFuncs<T> = SmallVec<[fn(T) -> T; 8]>;

/// Container of unary operators of one expression
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct UnaryOp<T> {
    funcs_to_be_composed: VecOfUnaryFuncs<T>,
}

impl<T> UnaryOp<T> {
    /// Applies unary operators one after the other starting with the last.
    /// # Arguments
    ///
    /// * `x` - number the unary operators are applied to
    ///
    pub fn apply(&self, x: T) -> T {
        let mut result = x;
        // rev, since the last uop is applied first by convention
        for uo in self.funcs_to_be_composed.iter().rev() {
            result = uo(result);
        }
        result
    }

    pub fn append_front(&mut self, other: &mut UnaryOp<T>) {
        self.funcs_to_be_composed = other
            .funcs_to_be_composed
            .iter()
            .chain(self.funcs_to_be_composed.iter())
            .map(|f| *f)
            .collect::<SmallVec<_>>();
    }

    pub fn len(&self) -> usize {
        self.funcs_to_be_composed.len()
    }

    pub fn new() -> Self {
        Self {
            funcs_to_be_composed: smallvec![],
        }
    }

    pub fn from_vec(v: VecOfUnaryFuncs<T>) -> Self {
        Self {
            funcs_to_be_composed: v,
        }
    }

    pub fn clear(&mut self) {
        self.funcs_to_be_composed.clear();
    }
}
