#![no_main]
use libfuzzer_sys::fuzz_target;
extern crate exmex;

use exmex::parse_with_default_ops;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data){
        let _ = parse_with_default_ops::<f64>(s);
    }
});
