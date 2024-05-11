#![no_main]
use libfuzzer_sys::fuzz_target;

use exmex::{DeepEx, prelude::*};

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let flatex = FlatEx::<f64>::parse(s);
        let _ = flatex.map(|x|x.to_deepex());
        let deepex = DeepEx::<f64>::parse(s);
        let _ = deepex.map(|d|FlatEx::from_deepex(d));
    }
});
