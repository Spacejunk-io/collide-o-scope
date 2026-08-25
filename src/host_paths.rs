//! Host filesystem conventions: where this process keeps its own state, and
//! where it finds the FFmpeg tools it shells out to.
//!
//! Both answers were previously spelled out at each call site, and both were
//! wrong off Windows in the same way: they assumed an environment the process
//! does not necessarily have. `%LOCALAPPDATA%` does not exist outside Windows,
//! and a macOS process launched from Finder inherits launchd's minimal `PATH`,
//! which contains neither `/opt/homebrew/bin` nor `/usr/local/bin`. Each site
//! then fell back to something relative — a state directory under the current
//! working directory, or a bare executable name the shell could not resolve —
//! so the failure was silent and looked like a missing feature rather than a
//! missing path.
//!
//! One law each, in one place, so the sixth caller cannot diverge again.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// State directory
// ---------------------------------------------------------------------------

/// The per-user directory this program keeps its own state in, resolved from
/// already-read environment values so the ladder itself stays testable.
///
/// Windows takes `%LOCALAPPDATA%`; every other host takes the XDG state
/// directory, then `~/.local/state`. The final relative fallback exists only
/// so a process with no usable environment still runs — it is a last resort,
/// never the expected answer.
pub fn state_root_from(
    local_app_data: Option<&OsStr>,
    xdg_state_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> PathBuf {
    if let Some(base) = local_app_data {
        return PathBuf::from(base).join("collide-o-scope");
    }
    if let Some(base) = xdg_state_home {
        return PathBuf::from(base).join("collide-o-scope");
    }
    if let Some(base) = home {
        return PathBuf::from(base)
            .join(".local")
            .join("state")
            .join("collide-o-scope");
    }
    PathBuf::from(".collide-o-scope")
}

/// [`state_root_from`] against this process's environment.
pub fn state_root() -> PathBuf {
    state_root_from(
        std::env::var_os("LOCALAPPDATA").as_deref(),
        std::env::var_os("XDG_STATE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}

// ---------------------------------------------------------------------------
// External FFmpeg tools
// ---------------------------------------------------------------------------

/// The two FFmpeg command-line tools this program runs as separate processes.
///
/// These are distinct from the FFmpeg *libraries* linked through
/// `ffmpeg-next`; the libraries are resolved by the dynamic loader and are not
/// affected by anything here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfmpegTool {
    Ffmpeg,
    Ffprobe,
}

impl FfmpegTool {
    /// The stem to look for, without any platform executable suffix.
    const fn stem(self) -> &'static str {
        match self {
            Self::Ffmpeg => "ffmpeg",
            Self::Ffprobe => "ffprobe",
        }
    }

    /// The environment variable that overrides discovery for this tool. An
    /// operator whose FFmpeg lives somewhere unusual sets this and is done.
    const fn override_var(self) -> &'static str {
        match self {
            Self::Ffmpeg => "COS_FFMPEG",
            Self::Ffprobe => "COS_FFPROBE",
        }
    }
}

/// Directories searched after `PATH` when a tool was not found on it.
///
/// Windows installers put FFmpeg on `PATH`, and `PATHEXT` resolution there is
/// reliable, so the list is deliberately empty; the fallback to the bare name
/// preserves the exact prior behaviour. The Unix entries are the Homebrew
/// prefixes for Apple Silicon and Intel, the usual system prefix, and MacPorts.
#[cfg(windows)]
const EXTRA_TOOL_DIRS: &[&str] = &[];
#[cfg(not(windows))]
const EXTRA_TOOL_DIRS: &[&str] = &[
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/opt/local/bin",
];

/// Candidate file names for a tool stem, most specific first.
fn candidate_file_names(stem: &str) -> Vec<OsString> {
    if cfg!(windows) {
        vec![OsString::from(format!("{stem}.exe")), OsString::from(stem)]
    } else {
        vec![OsString::from(stem)]
    }
}

/// Resolve one tool against already-read environment values.
///
/// Order: an explicit `COS_*` override, then `$FFMPEG_DIR/bin`, then `PATH`,
/// then the well-known install prefixes.
///
/// `is_file` is injected so the ladder can be exercised without touching the
/// filesystem. The final fallback is the bare stem, which is exactly what every
/// call site used before this module existed — so a host where discovery finds
/// nothing behaves no worse than it did, and a host where it finds something
/// behaves correctly.
fn resolve_tool_from(
    tool: FfmpegTool,
    override_value: Option<&OsStr>,
    ffmpeg_dir: Option<&OsStr>,
    path_var: Option<&OsStr>,
    extra_dirs: &[&str],
    is_file: &dyn Fn(&Path) -> bool,
) -> PathBuf {
    // An explicit override wins outright and is never probed: if an operator
    // names a path, a silent fallback to some other FFmpeg would be worse than
    // a clear failure to launch it.
    if let Some(value) = override_value {
        if !value.is_empty() {
            return PathBuf::from(value);
        }
    }

    let names = candidate_file_names(tool.stem());

    // A prefix the operator pinned for the *libraries* is searched before
    // `PATH`, because tools shipped beside those libraries are the ones whose
    // version matches what the program linked against.
    let pinned = ffmpeg_dir
        .filter(|value| !value.is_empty())
        .map(|value| PathBuf::from(value).join("bin"));
    let searched = path_var
        .map(|value| std::env::split_paths(value).collect::<Vec<_>>())
        .unwrap_or_default();

    for directory in pinned
        .iter()
        .map(PathBuf::as_path)
        .chain(searched.iter().map(PathBuf::as_path))
        .chain(extra_dirs.iter().map(Path::new))
    {
        if directory.as_os_str().is_empty() {
            continue;
        }
        for name in &names {
            let candidate = directory.join(name);
            if is_file(&candidate) {
                return candidate;
            }
        }
    }

    PathBuf::from(tool.stem())
}

fn resolve_tool(tool: FfmpegTool) -> PathBuf {
    resolve_tool_from(
        tool,
        std::env::var_os(tool.override_var()).as_deref(),
        std::env::var_os("FFMPEG_DIR").as_deref(),
        std::env::var_os("PATH").as_deref(),
        EXTRA_TOOL_DIRS,
        &|path| path.is_file(),
    )
}

/// The resolved `ffmpeg` executable, discovered once per process.
pub fn ffmpeg() -> &'static Path {
    static RESOLVED: OnceLock<PathBuf> = OnceLock::new();
    RESOLVED.get_or_init(|| resolve_tool(FfmpegTool::Ffmpeg))
}

/// The resolved `ffprobe` executable, discovered once per process.
pub fn ffprobe() -> &'static Path {
    static RESOLVED: OnceLock<PathBuf> = OnceLock::new();
    RESOLVED.get_or_init(|| resolve_tool(FfmpegTool::Ffprobe))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(value: &str) -> OsString {
        OsString::from(value)
    }

    #[test]
    fn state_root_prefers_local_app_data_then_xdg_then_home() {
        let local = os("C:/Users/x/AppData/Local");
        let xdg = os("/home/x/.state");
        let home = os("/home/x");

        assert_eq!(
            state_root_from(Some(&local), Some(&xdg), Some(&home)),
            PathBuf::from("C:/Users/x/AppData/Local").join("collide-o-scope"),
        );
        assert_eq!(
            state_root_from(None, Some(&xdg), Some(&home)),
            PathBuf::from("/home/x/.state").join("collide-o-scope"),
        );
        assert_eq!(
            state_root_from(None, None, Some(&home)),
            PathBuf::from("/home/x")
                .join(".local")
                .join("state")
                .join("collide-o-scope"),
        );
    }

    #[test]
    fn state_root_falls_back_to_a_relative_directory_with_no_environment() {
        assert_eq!(
            state_root_from(None, None, None),
            PathBuf::from(".collide-o-scope"),
        );
    }

    #[test]
    fn an_override_wins_outright_and_is_never_probed() {
        let override_value = os("/somewhere/custom/ffmpeg");
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            Some(&override_value),
            None,
            Some(&os("/usr/bin")),
            &["/opt/homebrew/bin"],
            // Nothing exists anywhere; the override must still be returned.
            &|_| false,
        );
        assert_eq!(resolved, PathBuf::from("/somewhere/custom/ffmpeg"));
    }

    #[test]
    fn an_empty_override_is_ignored_rather_than_resolving_to_nothing() {
        let empty = os("");
        let found = Path::new("/opt/homebrew/bin").join(&candidate_file_names("ffmpeg")[0]);
        let expected = found.clone();
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            Some(&empty),
            None,
            None,
            &["/opt/homebrew/bin"],
            &move |path| path == found,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn path_is_searched_before_the_well_known_directories() {
        let on_path = Path::new("/first/on/path").join(&candidate_file_names("ffprobe")[0]);
        let in_extra = Path::new("/opt/homebrew/bin").join(&candidate_file_names("ffprobe")[0]);
        let expected = on_path.clone();
        let resolved = resolve_tool_from(
            FfmpegTool::Ffprobe,
            None,
            None,
            Some(&os("/first/on/path")),
            &["/opt/homebrew/bin"],
            // Both exist; PATH must win.
            &move |path| path == on_path || path == in_extra,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn the_well_known_directories_rescue_a_minimal_launchd_path() {
        // This is the Finder-launched macOS case: PATH holds only the system
        // directories, and Homebrew's prefix is not among them.
        let homebrew = Path::new("/opt/homebrew/bin").join(&candidate_file_names("ffmpeg")[0]);
        let expected = homebrew.clone();
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            None,
            None,
            Some(&os("/usr/bin:/bin:/usr/sbin:/sbin")),
            &["/opt/homebrew/bin", "/usr/local/bin"],
            &move |path| path == homebrew,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn an_undiscoverable_tool_falls_back_to_the_bare_name() {
        // Preserves the exact behaviour every call site had before this module,
        // so discovery can only ever improve a host, never regress one.
        assert_eq!(
            resolve_tool_from(
                FfmpegTool::Ffmpeg,
                None,
                None,
                Some(&os("/nowhere")),
                &["/also/nowhere"],
                &|_| false,
            ),
            PathBuf::from("ffmpeg"),
        );
        assert_eq!(
            resolve_tool_from(FfmpegTool::Ffprobe, None, None, None, &[], &|_| false),
            PathBuf::from("ffprobe"),
        );
    }

    #[test]
    fn empty_path_entries_are_never_probed_as_the_working_directory() {
        // Some shells read an empty PATH entry as "the current directory".
        // Resolving a tool from the working directory would be surprising and a
        // security hazard, so those entries are dropped before probing.
        let probed: std::cell::RefCell<Vec<PathBuf>> = std::cell::RefCell::new(Vec::new());
        let path_var = os(if cfg!(windows) {
            ";C:/tools"
        } else {
            ":/tools"
        });
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            None,
            None,
            Some(&path_var),
            &[],
            &|path| {
                probed.borrow_mut().push(path.to_path_buf());
                false
            },
        );

        assert_eq!(resolved, PathBuf::from("ffmpeg"));
        assert!(
            !probed.borrow().is_empty(),
            "the real PATH entry should still have been probed",
        );
        for candidate in probed.borrow().iter() {
            let parent = candidate.parent().expect("candidate has a parent");
            assert!(
                !parent.as_os_str().is_empty(),
                "probed a working-directory candidate from an empty PATH entry: {}",
                candidate.display(),
            );
        }
    }

    #[test]
    fn a_pinned_ffmpeg_dir_is_searched_before_path() {
        // The libraries were linked from this prefix, so tools shipped beside
        // them are the version-matched ones.
        let pinned = Path::new("/pinned/9.0.1/bin").join(&candidate_file_names("ffmpeg")[0]);
        let on_path = Path::new("/usr/bin").join(&candidate_file_names("ffmpeg")[0]);
        let expected = pinned.clone();
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            None,
            Some(&os("/pinned/9.0.1")),
            Some(&os("/usr/bin")),
            &[],
            &move |path| path == pinned || path == on_path,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn a_pinned_prefix_without_the_tool_falls_through_to_path() {
        // The documented macOS recipe configures --disable-programs, so the
        // pinned prefix holds libraries and no command line at all.
        let on_path = Path::new("/usr/bin").join(&candidate_file_names("ffmpeg")[0]);
        let expected = on_path.clone();
        let resolved = resolve_tool_from(
            FfmpegTool::Ffmpeg,
            None,
            Some(&os("/pinned/9.0.1")),
            Some(&os("/usr/bin")),
            &[],
            &move |path| path == on_path,
        );
        assert_eq!(resolved, expected);
    }

    #[test]
    fn the_two_tools_have_distinct_stems_and_override_variables() {
        assert_eq!(FfmpegTool::Ffmpeg.stem(), "ffmpeg");
        assert_eq!(FfmpegTool::Ffprobe.stem(), "ffprobe");
        assert_eq!(FfmpegTool::Ffmpeg.override_var(), "COS_FFMPEG");
        assert_eq!(FfmpegTool::Ffprobe.override_var(), "COS_FFPROBE");
    }
}
