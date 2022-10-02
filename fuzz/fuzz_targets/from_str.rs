#![no_main]
use libfuzzer_sys::fuzz_target;

use exmex::{DeepEx, prelude::*};

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = FlatEx::<f64>::parse(s);
        let _ = DeepEx::<f64>::parse(s);
    }
});
