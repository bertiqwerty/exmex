
#![no_main]
use libfuzzer_sys::fuzz_target;

use exmex;

#[cfg(feature="value")]
fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data){
        let _ = exmex::line_2_statement_val::<i32, f64>(&s);
    }
});