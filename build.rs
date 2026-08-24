use std::env;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

mod build_identity_policy;

use build_identity_policy::{
    dotted_version, executable_basename, git_sha as normalize_git_sha, git_status_output_is_dirty,
    rustc_verbose_version, safe_fact, tool_status_is_accepted, tool_version_line,
};

const BUILD_IDENTITY_DOMAIN: &str = "collide-o-scope build identity v1";
const SHADER_BUNDLE_DOMAIN: &[u8] = b"collide-o-scope shader bundle v1\0";

#[derive(Debug)]
struct GeneratedIdentity {
    package_name: String,
    version: String,
    git_sha: String,
    git_dirty: bool,
    profile: String,
    target: String,
    enabled_features: String,
    rustc_vv: String,
    cargo_version: String,
    linker_identity: String,
    sdk_identity: String,
    ffmpeg_libraries: String,
    ffmpeg_binary_version: String,
    ffmpeg_binary_sha256: String,
    ffprobe_binary_version: String,
    ffprobe_binary_sha256: String,
    shader_bundle_sha256: String,
    cargo_lock_sha256: String,
    published_artifact: bool,
}

impl GeneratedIdentity {
    fn canonical_payload(&self) -> String {
        let mut payload = String::new();
        for (key, value) in [
            ("domain", BUILD_IDENTITY_DOMAIN.to_owned()),
            ("package_name", self.package_name.clone()),
            ("version", self.version.clone()),
            ("git_sha", self.git_sha.clone()),
            ("git_dirty", self.git_dirty.to_string()),
            ("profile", self.profile.clone()),
            ("target", self.target.clone()),
            ("enabled_features", self.enabled_features.clone()),
            ("rustc_vv", self.rustc_vv.clone()),
            ("cargo_version", self.cargo_version.clone()),
            ("linker_identity", self.linker_identity.clone()),
            ("sdk_identity", self.sdk_identity.clone()),
            ("ffmpeg_libraries", self.ffmpeg_libraries.clone()),
            ("ffmpeg_binary_version", self.ffmpeg_binary_version.clone()),
            ("ffmpeg_binary_sha256", self.ffmpeg_binary_sha256.clone()),
            (
                "ffprobe_binary_version",
                self.ffprobe_binary_version.clone(),
            ),
            ("ffprobe_binary_sha256", self.ffprobe_binary_sha256.clone()),
            ("shader_bundle_sha256", self.shader_bundle_sha256.clone()),
            ("cargo_lock_sha256", self.cargo_lock_sha256.clone()),
            ("published_artifact", self.published_artifact.to_string()),
        ] {
            let _ = writeln!(payload, "{key}={value}");
        }
        payload
    }
}

fn normalized_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .trim()
        .chars()
        .take(16 * 1024)
        .collect()
}

fn command_output_allowing(
    program: impl AsRef<OsStr>,
    args: &[&str],
    reviewed_nonzero: &[i32],
) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !tool_status_is_accepted(
        output.status.success(),
        output.status.code(),
        reviewed_nonzero,
    ) {
        return None;
    }
    let stdout = normalized_text(&output.stdout);
    let stderr = normalized_text(&output.stderr);
    match (stdout.is_empty(), stderr.is_empty()) {
        (false, true) => Some(stdout),
        (true, false) => Some(stderr),
        (false, false) => Some(format!("{stdout}\n{stderr}")),
        (true, true) => None,
    }
}

fn command_output(program: impl AsRef<OsStr>, args: &[&str]) -> Option<String> {
    command_output_allowing(program, args, &[])
}

fn env_override(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.replace("\r\n", "\n").trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn git_output(manifest_dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(manifest_dir)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| normalized_text(&output.stdout))
        .filter(|value| !value.is_empty())
}

fn git_dirty(manifest_dir: &Path) -> bool {
    let observed = Command::new("git")
        .current_dir(manifest_dir)
        .args(["status", "--porcelain=v1", "--untracked-files=normal"])
        .output()
        .map_or(true, |output| {
            git_status_output_is_dirty(output.status.success(), &output.stdout)
        });
    match env_override("COLLIDE_BUILD_GIT_DIRTY").as_deref() {
        Some("1" | "true" | "yes") => true,
        // A caller may preserve a known-dirty fact, but it cannot use an
        // environment override to launder an observed dirty tree as clean.
        Some("0" | "false" | "no") | None => observed,
        Some(_) => true,
    }
}

fn sha256_file(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn shader_bundle_sha256(manifest_dir: &Path) -> String {
    let shader_dir = manifest_dir.join("src").join("shaders");
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    let mut shaders = fs::read_dir(&shader_dir)
        .unwrap_or_else(|error| panic!("read shader directory {}: {error}", shader_dir.display()))
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension() == Some(OsStr::new("wgsl")))
        .collect::<Vec<_>>();
    shaders.sort();
    let mut digest = Sha256::new();
    digest.update(SHADER_BUNDLE_DOMAIN);
    for path in shaders {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path
            .strip_prefix(manifest_dir)
            .expect("shader is below manifest")
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("read embedded shader {}: {error}", path.display()));
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

fn enabled_features() -> String {
    let mut features = env::vars_os()
        .filter_map(|(key, _)| {
            key.to_str()
                .and_then(|key| key.strip_prefix("CARGO_FEATURE_"))
                .map(|feature| feature.to_ascii_lowercase().replace('_', "-"))
        })
        .collect::<Vec<_>>();
    features.sort();
    features.dedup();
    if features.is_empty() {
        "(none)".to_owned()
    } else {
        features.join(",")
    }
}

fn sdk_identity() -> String {
    let mut parts = Vec::new();
    for (name, label) in [
        ("WindowsSDKVersion", "windows-sdk"),
        ("VCToolsVersion", "msvc-tools"),
        ("MACOSX_DEPLOYMENT_TARGET", "macos-deployment"),
    ] {
        if let Some(value) = env_override(name).and_then(|value| dotted_version(&value)) {
            parts.push(format!("{label}:{value}"));
        }
    }
    if parts.is_empty() {
        "unreported".to_owned()
    } else {
        parts.join(";")
    }
}

fn linker_identity(target: &str) -> String {
    let target_key = format!(
        "CARGO_TARGET_{}_LINKER",
        target.to_ascii_uppercase().replace(['-', '.'], "_")
    );
    println!("cargo:rerun-if-env-changed={target_key}");
    let configured = env_override(&target_key).or_else(|| env_override("RUSTC_LINKER"));
    let program = configured.unwrap_or_else(|| {
        if target.contains("windows-msvc") {
            "link.exe".to_owned()
        } else {
            "cc".to_owned()
        }
    });
    let Some(basename) = executable_basename(&program) else {
        return "invalid-linker-identity".to_owned();
    };
    let is_msvc_link = basename.eq_ignore_ascii_case("link.exe") && target.contains("windows-msvc");
    let arguments: &[&str] = if is_msvc_link {
        &["/?"]
    } else {
        &["--version"]
    };
    let prefixes: &[&str] = if is_msvc_link {
        &["Microsoft (R) Incremental Linker Version"]
    } else {
        &[
            "cc ",
            "clang version ",
            "Apple clang version ",
            "gcc ",
            "GNU ld ",
            "LLD ",
            "mold ",
        ]
    };
    let reviewed_nonzero: &[i32] = if is_msvc_link { &[1100] } else { &[] };
    command_output_allowing(&program, arguments, reviewed_nonzero)
        .and_then(|output| tool_version_line(&output, prefixes))
        .map(|version| format!("{basename};{version}"))
        .and_then(|identity| safe_fact(&identity, 640))
        .unwrap_or(basename)
}

fn ffmpeg_library_identity(ffmpeg_dir: Option<&Path>) -> String {
    let mut parts = Vec::new();
    if let Some(version) = env_override("FFMPEG_VERSION").and_then(|value| dotted_version(&value)) {
        parts.push(format!("ffmpeg={version}"));
    }
    if let Some(root) = ffmpeg_dir {
        for directory in [root.join("bin"), root.join("lib")] {
            let Ok(entries) = fs::read_dir(directory) else {
                continue;
            };
            for entry in entries.filter_map(Result::ok) {
                let name = entry.file_name().to_string_lossy().to_string();
                let lower = name.to_ascii_lowercase();
                if [
                    "avcodec",
                    "avdevice",
                    "avfilter",
                    "avformat",
                    "avutil",
                    "swscale",
                    "swresample",
                ]
                .iter()
                .any(|stem| lower.starts_with(stem) || lower.starts_with(&format!("lib{stem}")))
                    && (lower.ends_with(".dll")
                        || lower.contains(".so")
                        || lower.contains(".dylib"))
                {
                    if let Some(name) = safe_fact(&name, 192) {
                        parts.push(name);
                    }
                }
            }
        }
    }
    parts.sort();
    parts.dedup();
    if parts.is_empty() {
        "unreported".to_owned()
    } else {
        parts.join(",")
    }
}

fn find_ffmpeg_tool(ffmpeg_dir: Option<&Path>, stem: &str) -> Option<PathBuf> {
    let override_name = match stem {
        "ffmpeg" => "COLLIDE_BUILD_FFMPEG_BINARY",
        "ffprobe" => "COLLIDE_BUILD_FFPROBE_BINARY",
        _ => return None,
    };
    if let Some(path) = env::var_os(override_name).map(PathBuf::from) {
        return path.is_file().then_some(path);
    }
    let root = ffmpeg_dir?;
    let candidate = root.join("bin").join(if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_owned()
    });
    candidate.is_file().then_some(candidate)
}

fn ffmpeg_tool_identity(path: Option<&Path>) -> (String, String) {
    let Some(path) = path else {
        return (
            "external-or-unavailable".to_owned(),
            "unreported".to_owned(),
        );
    };
    let stem = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
    let expected_prefix = if stem.eq_ignore_ascii_case("ffprobe") {
        "ffprobe version "
    } else {
        "ffmpeg version "
    };
    let version = command_output(path.as_os_str(), &["-version"])
        .and_then(|output| tool_version_line(&output, &[expected_prefix]))
        .unwrap_or_else(|| "unreported".to_owned());
    let sha256 = sha256_file(path).unwrap_or_else(|| "unreported".to_owned());
    (version, sha256)
}

fn rust_literal(value: &str) -> String {
    format!("{value:?}")
}

fn write_generated_identity(out_dir: &Path, identity: &GeneratedIdentity) {
    let payload = identity.canonical_payload();
    let identity_sha256 = format!("{:x}", Sha256::digest(payload.as_bytes()));
    let generated = format!(
        "pub const GENERATED_BUILD_IDENTITY: BuildIdentity = BuildIdentity {{\n\
         schema_version: 1,\n\
         package_name: {},\n\
         version: {},\n\
         git_sha: {},\n\
         git_dirty: {},\n\
         profile: {},\n\
         target: {},\n\
         enabled_features: {},\n\
         rustc_vv: {},\n\
         cargo_version: {},\n\
         linker_identity: {},\n\
         sdk_identity: {},\n\
         ffmpeg_libraries: {},\n\
         ffmpeg_binary_version: {},\n\
         ffmpeg_binary_sha256: {},\n\
         ffprobe_binary_version: {},\n\
         ffprobe_binary_sha256: {},\n\
         shader_bundle_sha256: {},\n\
         cargo_lock_sha256: {},\n\
         identity_sha256: {},\n\
         published_artifact: {},\n\
         }};\n",
        rust_literal(&identity.package_name),
        rust_literal(&identity.version),
        rust_literal(&identity.git_sha),
        identity.git_dirty,
        rust_literal(&identity.profile),
        rust_literal(&identity.target),
        rust_literal(&identity.enabled_features),
        rust_literal(&identity.rustc_vv),
        rust_literal(&identity.cargo_version),
        rust_literal(&identity.linker_identity),
        rust_literal(&identity.sdk_identity),
        rust_literal(&identity.ffmpeg_libraries),
        rust_literal(&identity.ffmpeg_binary_version),
        rust_literal(&identity.ffmpeg_binary_sha256),
        rust_literal(&identity.ffprobe_binary_version),
        rust_literal(&identity.ffprobe_binary_sha256),
        rust_literal(&identity.shader_bundle_sha256),
        rust_literal(&identity.cargo_lock_sha256),
        rust_literal(&identity_sha256),
        identity.published_artifact,
    );
    fs::write(out_dir.join("build_identity.rs"), generated)
        .expect("write generated build identity");
}

fn emit_build_identity() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_owned());
    let observed_git_sha = git_output(&manifest_dir, &["rev-parse", "HEAD"])
        .and_then(|value| normalize_git_sha(&value));
    let requested_git_sha =
        env_override("COLLIDE_BUILD_GIT_SHA").and_then(|value| normalize_git_sha(&value));
    let git_sha = requested_git_sha
        .clone()
        .or_else(|| observed_git_sha.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let git_dirty = git_dirty(&manifest_dir);
    let published_artifact = env_override("COLLIDE_PUBLISHED_ARTIFACT")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    if published_artifact
        && (git_dirty
            || git_sha == "unknown"
            || observed_git_sha.as_deref() != Some(git_sha.as_str()))
    {
        panic!(
            "a published-artifact build requires a clean checkout at the exact embedded Git SHA"
        );
    }

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let ffmpeg_dir = env::var_os("FFMPEG_DIR").map(PathBuf::from);
    let ffmpeg = find_ffmpeg_tool(ffmpeg_dir.as_deref(), "ffmpeg");
    let ffprobe = find_ffmpeg_tool(ffmpeg_dir.as_deref(), "ffprobe");
    let (ffmpeg_binary_version, ffmpeg_binary_sha256) = ffmpeg_tool_identity(ffmpeg.as_deref());
    let (ffprobe_binary_version, ffprobe_binary_sha256) = ffmpeg_tool_identity(ffprobe.as_deref());
    let cargo_lock = manifest_dir.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", cargo_lock.display());
    println!("cargo:rerun-if-env-changed=COLLIDE_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=COLLIDE_BUILD_GIT_DIRTY");
    println!("cargo:rerun-if-env-changed=RUSTC_LINKER");
    println!("cargo:rerun-if-env-changed=WindowsSDKVersion");
    println!("cargo:rerun-if-env-changed=VCToolsVersion");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=COLLIDE_BUILD_FFMPEG_BINARY");
    println!("cargo:rerun-if-env-changed=COLLIDE_BUILD_FFPROBE_BINARY");
    println!("cargo:rerun-if-env-changed=COLLIDE_PUBLISHED_ARTIFACT");
    println!("cargo:rerun-if-env-changed=FFMPEG_DIR");
    println!("cargo:rerun-if-env-changed=FFMPEG_VERSION");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    let identity = GeneratedIdentity {
        package_name: env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "collide-o-scope".into()),
        version: env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into()),
        git_sha,
        git_dirty,
        profile: env::var("PROFILE").unwrap_or_else(|_| "unknown".into()),
        target: target.clone(),
        enabled_features: enabled_features(),
        rustc_vv: command_output(rustc, &["-Vv"])
            .and_then(|output| rustc_verbose_version(&output))
            .unwrap_or_else(|| "unreported".to_owned()),
        cargo_version: command_output(cargo, &["-V"])
            .and_then(|output| tool_version_line(&output, &["cargo "]))
            .unwrap_or_else(|| "unreported".to_owned()),
        linker_identity: linker_identity(&target),
        sdk_identity: sdk_identity(),
        ffmpeg_libraries: ffmpeg_library_identity(ffmpeg_dir.as_deref()),
        ffmpeg_binary_version,
        ffmpeg_binary_sha256,
        ffprobe_binary_version,
        ffprobe_binary_sha256,
        shader_bundle_sha256: shader_bundle_sha256(&manifest_dir),
        cargo_lock_sha256: sha256_file(&cargo_lock).unwrap_or_else(|| "unreported".to_owned()),
        published_artifact,
    };
    if published_artifact
        && (identity.rustc_vv == "unreported"
            || identity.cargo_version == "unreported"
            || !identity.linker_identity.contains(';')
            || identity.sdk_identity == "unreported"
            || identity.ffmpeg_libraries == "unreported"
            || !identity
                .ffmpeg_binary_version
                .starts_with("ffmpeg version ")
            || identity.ffmpeg_binary_sha256 == "unreported"
            || !identity
                .ffprobe_binary_version
                .starts_with("ffprobe version ")
            || identity.ffprobe_binary_sha256 == "unreported")
    {
        panic!("a published-artifact build requires complete compiler, SDK, and FFmpeg identity");
    }
    write_generated_identity(&out_dir, &identity);
}

fn main() {
    emit_build_identity();

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
