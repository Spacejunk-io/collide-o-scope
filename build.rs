fn main() {
    // The vendored Spout2 SDK calls TaskDialogIndirect (ComCtl32 ordinal 345),
    // which only exists in ComCtl32 v6 — and v6 is only loaded when the exe's
    // manifest declares the dependency. Rust doesn't embed one by default, so
    // without this the loader binds System32's ComCtl32 5.82 and the process
    // dies at startup with STATUS_ORDINAL_NOT_FOUND before main() runs.
    #[cfg(target_env = "msvc")]
    {
        println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
        println!(
            "cargo:rustc-link-arg-bins=/MANIFESTDEPENDENCY:type='win32' \
             name='Microsoft.Windows.Common-Controls' version='6.0.0.0' \
             processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'"
        );
    }
}
