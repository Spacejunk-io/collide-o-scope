fn main() {
    // The shell reads an executable's icon from its PE resources. winit's
    // `with_window_icon` covers the title bar and alt-tab at runtime, but the
    // taskbar button and Explorer both want the embedded resource, so the
    // program ships both. A failure here is cosmetic and must never fail a
    // build, so it degrades to a warning.
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/icon/collide-o-scope.ico");
        if let Err(error) = resource.compile() {
            println!("cargo:warning=program icon not embedded: {error}");
        }
    }
    println!("cargo:rerun-if-changed=assets/icon/collide-o-scope.ico");

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
