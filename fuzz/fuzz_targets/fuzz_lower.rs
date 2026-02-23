#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(source) = std::str::from_utf8(data) {
        // The full parse + lower pipeline should never panic.
        let _ = nxpu_parser::parse(source);
    }
});
