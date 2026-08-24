//! Path-free normalization used by both `build.rs` and hostile-input tests.

pub const MAX_IDENTITY_FACT_BYTES: usize = 2_048;

/// Interpret the exact stdout contract of `git status --porcelain`.
///
/// A successful command with no bytes is the only clean result. Command
/// failure and every emitted byte fail closed as dirty without depending on
/// UTF-8 decoding or conflating empty output with an unavailable command.
pub fn git_status_output_is_dirty(command_succeeded: bool, stdout: &[u8]) -> bool {
    !command_succeeded || !stdout.is_empty()
}

/// Admit a tool result only when it succeeded or returned one explicitly
/// reviewed nonzero status. This is used for MSVC `link.exe /?`, whose version
/// banner is valid while the process deliberately exits with code 1100.
pub fn tool_status_is_accepted(
    command_succeeded: bool,
    status_code: Option<i32>,
    reviewed_nonzero: &[i32],
) -> bool {
    command_succeeded || status_code.is_some_and(|status| reviewed_nonzero.contains(&status))
}

fn safe_ascii(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            ' ' | '.' | '_' | '-' | '+' | '=' | ':' | ';' | ',' | '(' | ')' | '[' | ']' | '@'
        )
}

pub fn safe_fact(value: &str, max_bytes: usize) -> Option<String> {
    let value = value.trim();
    if value.is_empty()
        || matches!(value, "." | "..")
        || value.len() > max_bytes.min(MAX_IDENTITY_FACT_BYTES)
        || value.contains(['/', '\\'])
        || !value.chars().all(safe_ascii)
    {
        return None;
    }
    Some(value.to_owned())
}

pub fn executable_basename(value: &str) -> Option<String> {
    let basename = value
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty() && !matches!(*name, "." | ".."))?;
    safe_fact(basename, 128)
}

pub fn git_sha(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    matches!(value.len(), 40 | 64)
        .then_some(())
        .filter(|_| value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .map(|_| value)
}

pub fn dotted_version(value: &str) -> Option<String> {
    let value = value.trim().trim_end_matches(['/', '\\']);
    if value.is_empty()
        || value.len() > 64
        || !value.starts_with(|character: char| character.is_ascii_digit())
        || !value.ends_with(|character: char| character.is_ascii_digit())
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'.' | b'-' | b'_'))
    {
        return None;
    }
    Some(value.to_owned())
}

fn numeric_version_from_text(value: &str) -> Option<String> {
    value.split_ascii_whitespace().find_map(|token| {
        if !token.starts_with(|character: char| character.is_ascii_digit()) {
            return None;
        }
        let candidate = token
            .chars()
            .take_while(|character| {
                character.is_ascii_digit() || matches!(character, '.' | '-' | '_')
            })
            .collect::<String>()
            .trim_end_matches(['.', '-', '_'])
            .to_owned();
        dotted_version(&candidate)
    })
}

pub fn tool_version_line(output: &str, prefixes: &[&str]) -> Option<String> {
    output.lines().find_map(|line| {
        let line = line.trim();
        prefixes.iter().find_map(|prefix| {
            let suffix = line.strip_prefix(prefix)?;
            let version = numeric_version_from_text(suffix)?;
            safe_fact(&format!("{} {version}", prefix.trim()), 192)
        })
    })
}

pub fn rustc_verbose_version(output: &str) -> Option<String> {
    fn value<'a>(output: &'a str, prefix: &str) -> Option<&'a str> {
        output
            .lines()
            .find_map(|line| line.trim().strip_prefix(prefix))
            .map(str::trim)
    }

    let headline = tool_version_line(output, &["rustc "])?;
    let binary = executable_basename(value(output, "binary: ")?)?;
    if !matches!(binary.as_str(), "rustc" | "rustc.exe") {
        return None;
    }
    let commit_hash = git_sha(value(output, "commit-hash: ")?)?;
    let commit_date = value(output, "commit-date: ")?;
    if commit_date.len() != 10
        || !commit_date.bytes().enumerate().all(|(index, byte)| {
            matches!(index, 4 | 7) && byte == b'-'
                || !matches!(index, 4 | 7) && byte.is_ascii_digit()
        })
    {
        return None;
    }
    let host = value(output, "host: ")?;
    if host.is_empty()
        || host.len() > 128
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return None;
    }
    let release = dotted_version(value(output, "release: ")?)?;
    let llvm = dotted_version(value(output, "LLVM version: ")?)?;
    Some(
        [
            headline,
            format!("binary: {binary}"),
            format!("commit-hash: {commit_hash}"),
            format!("commit-date: {commit_date}"),
            format!("host: {host}"),
            format!("release: {release}"),
            format!("LLVM version: {llvm}"),
        ]
        .join("\n"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn porcelain_status_distinguishes_clean_output_from_command_failure() {
        assert!(!git_status_output_is_dirty(true, b""));
        assert!(git_status_output_is_dirty(true, b"?? untracked\n"));
        assert!(git_status_output_is_dirty(true, b"\xff"));
        assert!(git_status_output_is_dirty(false, b""));
    }

    #[test]
    fn only_an_explicitly_reviewed_nonzero_tool_status_is_accepted() {
        assert!(tool_status_is_accepted(true, Some(0), &[]));
        assert!(tool_status_is_accepted(false, Some(1100), &[1100]));
        assert!(!tool_status_is_accepted(false, Some(1100), &[]));
        assert!(!tool_status_is_accepted(false, Some(1), &[1100]));
        assert!(!tool_status_is_accepted(false, None, &[1100]));
    }

    #[test]
    fn msvc_linker_banner_is_reduced_to_the_closed_version_fact() {
        assert_eq!(
            tool_version_line(
                "Microsoft (R) Incremental Linker Version 14.44.35228.0\n\
Copyright (C) Microsoft Corporation. All rights reserved.\n",
                &["Microsoft (R) Incremental Linker Version"],
            ),
            Some("Microsoft (R) Incremental Linker Version 14.44.35228.0".into())
        );
    }

    #[test]
    fn paths_controls_and_seeded_secret_shapes_are_rejected() {
        for hostile in [
            "C:\\agent\\_work\\secret-link.exe",
            "/opt/token/secret-link",
            "link.exe\nAUTH_TOKEN=seeded-secret",
            "link.exe\u{1b}[31m",
            "..",
        ] {
            assert_eq!(safe_fact(hostile, 256), None);
        }
        assert_eq!(
            executable_basename("C:\\toolchain\\bin\\link.exe"),
            Some("link.exe".into())
        );
        assert_eq!(executable_basename("/usr/bin/clang"), Some("clang".into()));
    }

    #[test]
    fn git_and_sdk_versions_are_typed_not_arbitrary_environment_text() {
        assert_eq!(git_sha(&"a".repeat(40)), Some("a".repeat(40)));
        assert_eq!(git_sha("not-a-sha"), None);
        assert_eq!(
            dotted_version("10.0.26100.0\\"),
            Some("10.0.26100.0".into())
        );
        assert_eq!(dotted_version("10.0;TOKEN=seeded"), None);
    }

    #[test]
    fn rustc_identity_keeps_only_the_closed_version_vocabulary() {
        let input = "rustc 1.98.0 (abc 2026-08-01)\n\
binary: rustc\n\
commit-hash: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
commit-date: 2026-08-01\n\
host: x86_64-pc-windows-msvc\n\
release: 1.98.0\n\
LLVM version: 21.1.0\n\
AUTH_TOKEN: should-not-survive\n";
        let normalized = rustc_verbose_version(input).unwrap();
        assert!(!normalized.contains("AUTH_TOKEN"));
        assert!(!normalized.contains("should-not-survive"));
        assert_eq!(normalized.lines().count(), 7);
        assert_eq!(normalized.lines().next(), Some("rustc 1.98.0"));
    }

    #[test]
    fn tool_output_must_use_an_expected_path_free_prefix() {
        assert_eq!(
            tool_version_line(
                "noise\nffmpeg version 8.1.2 Copyright",
                &["ffmpeg version "]
            ),
            Some("ffmpeg version 8.1.2".into())
        );
        assert_eq!(
            tool_version_line(
                "ffmpeg version 8.1.2-full_build-www.gyan.dev Copyright (c) 2000-2026",
                &["ffmpeg version "]
            ),
            Some("ffmpeg version 8.1.2".into())
        );
        assert_eq!(
            tool_version_line("ffmpeg version C:\\secret\\build", &["ffmpeg version "]),
            None
        );
        assert_eq!(
            tool_version_line(
                "ffmpeg version 8.1.2 seeded-auth-token authored text",
                &["ffmpeg version "]
            ),
            Some("ffmpeg version 8.1.2".into())
        );
    }
}
