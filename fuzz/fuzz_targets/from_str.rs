#![no_main]
use libfuzzer_sys::fuzz_target;

use exmex::prelude::*;

fuzz_target!(|data: &[u8]| {
    let dummy = [0u8];
    let split = if data.len() > 1 {
        data.len() / 2
    } else {
        0
    };
    let (d1, d2) = if split > 0 {
        (&data[..split], &data[split..])    
    } else {        
        (data, &dummy[0..1])
    };
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = FlatEx::<f64>::from_str(s);
        let _ = FlatEx::<f64>::from_pattern(s, s);
    }
    if let (Ok(s1), Ok(s2)) = (std::str::from_utf8(d1), std::str::from_utf8(d2)) {
        let _ = FlatEx::<f64>::from_pattern(s1, s2);
    }
});
