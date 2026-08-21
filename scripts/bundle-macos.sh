#!/usr/bin/env bash
#
# Assemble collide-o-scope.app for macOS.
#
# A bare Mach-O binary has nowhere to carry usage-description strings, so macOS
# denies Local Network and microphone access outright instead of prompting. That
# silently breaks the browser control panel and live audio analysis, and it is
# the reason this bundle exists rather than shipping the raw executable.
#
# The bundle is unsigned. Gatekeeper will quarantine it on first launch; either
# right-click -> Open once, or sign it yourself with your own identity.
#
# Usage:
#   scripts/bundle-macos.sh [--debug] [--output DIR]
#
# Environment:
#   FFMPEG_DIR   Prefix of the FFmpeg 8 build the program was linked against.
#                When set, its dylibs are copied into the bundle and the binary
#                is repointed at them, so the app runs from Finder without any
#                DYLD_* variable. See the macOS section of README.md.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "error: this script assembles a macOS app bundle and must run on macOS" >&2
	exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

profile="release"
cargo_profile_args=(--release)
output_dir=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--debug)
		profile="debug"
		cargo_profile_args=()
		shift
		;;
	--output)
		output_dir="${2:?--output needs a directory}"
		shift 2
		;;
	-h | --help)
		sed -n '3,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
		exit 0
		;;
	*)
		echo "error: unknown argument: $1" >&2
		exit 1
		;;
	esac
done

[[ -n "$output_dir" ]] || output_dir="target/macos"

binary="target/$profile/collide-o-scope"
app="$output_dir/collide-o-scope.app"
macos_dir="$app/Contents/MacOS"
resources_dir="$app/Contents/Resources"
frameworks_dir="$app/Contents/Frameworks"

echo "==> building ($profile)"
cargo build "${cargo_profile_args[@]}"
[[ -f "$binary" ]] || {
	echo "error: expected binary at $binary" >&2
	exit 1
}

echo "==> assembling $app"
rm -rf "$app"
mkdir -p "$macos_dir" "$resources_dir"
cp "$binary" "$macos_dir/collide-o-scope"
cp packaging/macos/Info.plist "$app/Contents/Info.plist"
printf 'APPL????' >"$app/Contents/PkgInfo"

# Deliberately NOT shipped: developer documentation of any kind. Contributor
# instructions are not program data — nothing reads them at build or run time,
# they are embedded in no binary and hashed into no receipt, and the program is
# fully functional without them. Copying them here would ship the project's
# internal working notes to every operator. If a future change needs a document
# inside the bundle, add that document explicitly; do not sweep the repository
# root.

# ---------------------------------------------------------------------------
# Icon
#
# The repository carries PNGs rather than a generated .icns, so the iconset is
# built here from the same source art the Windows icon uses. `sips` and
# `iconutil` are both part of macOS.
# ---------------------------------------------------------------------------
echo "==> icon"
iconset="$(mktemp -d)/collide-o-scope.iconset"
mkdir -p "$iconset"
source_icon="assets/icon/collide-o-scope-256.png"
for spec in "16:16x16" "32:16x16@2x" "32:32x32" "64:32x32@2x" "128:128x128" "256:128x128@2x" "256:256x256" "512:256x256@2x"; do
	size="${spec%%:*}"
	name="${spec##*:}"
	sips --resampleHeightWidth "$size" "$size" "$source_icon" \
		--out "$iconset/icon_$name.png" >/dev/null
done
iconutil --convert icns "$iconset" --output "$resources_dir/collide-o-scope.icns"

# ---------------------------------------------------------------------------
# FFmpeg libraries
#
# `ffmpeg-next` links the FFmpeg 8 shared libraries by their install names,
# which point at the prefix they were built in. A Finder launch inherits none of
# the DYLD_* variables that make that prefix findable, so the dylibs are copied
# in and every reference — the binary's, and the libraries' references to each
# other — is rewritten to @rpath.
#
# VERIFY ON HARDWARE: this is the one step in this script that cannot be
# checked without a Mac. Confirm with `otool -L` on the bundled binary that no
# absolute path into the build prefix survives, then launch from Finder.
# ---------------------------------------------------------------------------
if [[ -n "${FFMPEG_DIR:-}" && -d "${FFMPEG_DIR}/lib" ]]; then
	echo "==> vendoring FFmpeg dylibs from $FFMPEG_DIR"
	mkdir -p "$frameworks_dir"

	shopt -s nullglob
	dylibs=("$FFMPEG_DIR"/lib/*.dylib)
	shopt -u nullglob
	if [[ ${#dylibs[@]} -eq 0 ]]; then
		echo "error: no dylibs under $FFMPEG_DIR/lib" >&2
		exit 1
	fi

	# Copy real files, not the version symlinks, then recreate the names the
	# install names actually reference.
	for lib in "${dylibs[@]}"; do
		cp -a "$lib" "$frameworks_dir/"
	done
	chmod u+w "$frameworks_dir"/*.dylib

	install_name_tool -add_rpath "@executable_path/../Frameworks" \
		"$macos_dir/collide-o-scope" 2>/dev/null || true

	# Rewrite the binary's references, and each library's own id and references.
	rewrite_references() {
		local target="$1"
		local dependency base
		while read -r dependency; do
			case "$dependency" in
			"$FFMPEG_DIR"/lib/*)
				base="$(basename "$dependency")"
				install_name_tool -change "$dependency" "@rpath/$base" "$target"
				;;
			esac
		done < <(otool -L "$target" | tail -n +2 | awk '{print $1}')
	}

	rewrite_references "$macos_dir/collide-o-scope"
	for lib in "$frameworks_dir"/*.dylib; do
		install_name_tool -id "@rpath/$(basename "$lib")" "$lib"
		rewrite_references "$lib"
	done

	if otool -L "$macos_dir/collide-o-scope" | grep -q "$FFMPEG_DIR"; then
		echo "error: absolute FFmpeg paths survive in the bundled binary" >&2
		otool -L "$macos_dir/collide-o-scope" | grep "$FFMPEG_DIR" >&2
		exit 1
	fi
else
	echo "==> FFMPEG_DIR unset; not vendoring FFmpeg libraries"
	echo "    The app will only launch where the FFmpeg 8 dylibs are already"
	echo "    on the default search path."
fi

# The command-line tools are looked up at runtime, not bundled: they are a
# separate program under their own licence, and the operator may legitimately
# want a different build than the one the libraries came from. Point the app at
# an unusual install with COS_FFMPEG / COS_FFPROBE.
echo
echo "built $app"
echo
echo "next:"
echo "  open $app"
echo "  # first launch is quarantined; right-click -> Open, or sign the bundle"
