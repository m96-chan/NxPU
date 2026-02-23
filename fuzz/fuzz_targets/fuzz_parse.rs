#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(source) = std::str::from_utf8(data) {
        // naga's WGSL parser should never panic on any input.
        let _ = naga::front::wgsl::parse_str(source);
    }
});
