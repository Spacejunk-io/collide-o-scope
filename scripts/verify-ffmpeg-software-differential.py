#!/usr/bin/env python3
"""Compare bounded software decoding across version-pinned FFmpeg 8 and 9 SDKs.

The baseline FFmpeg alone creates a small deterministic fixture matrix in a
fresh temporary directory.  Both SDKs then decode and probe those exact bytes.
The verifier fails closed unless frame PTS/duration/SHA-256 sequences and a
stable subset of stream/frame metadata agree exactly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time


BASELINE_VERSION = "8.1.2"
CANDIDATE_VERSION = "9.0.1"
COMMAND_TIMEOUT_SECONDS = 45
MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_FIXTURE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_FIXTURE_BYTES = 24 * 1024 * 1024
MAX_TOTAL_WORK_BYTES = 32 * 1024 * 1024
MAX_FRAMES_PER_CASE = 32
MAX_COMMANDS = 64

WINDOWS_GYAN_IDENTITIES = {
    "baseline": {
        "version": BASELINE_VERSION,
        "distribution_version": "8.1.2-full_build-www.gyan.dev",
        "configuration_sha256": "6552e69554dab6a6d0977743fde1b8340700804f707c0b19d620cdc1b7f68ad6",
        "ffmpeg_sha256": "9be30133edcc5786f16e632d57e2dab5b259fea9a5243ba99758ca3adc110bd9",
        "ffprobe_sha256": "8fd54d7fc602ec180d023a5396f9d45d1a3d6f43a4c18d3217a1f0e306885fc9",
        "runtime_libraries": {
            "avcodec-62.dll": "34f5b1baac01c4be3edf464309c79db05ffbd4a9c905c94b4a4651cd15370296",
            "avdevice-62.dll": "d213d6cad9f3a526f7664ebb3f93d6882669540db7164daecd03f52e0f5288cc",
            "avfilter-11.dll": "e318cac83d648869180d0b57c45f21aad1e4db34a467000c157da2938ff7f63f",
            "avformat-62.dll": "c04e6ed2f9f36d42325d4f4df5babb5d6ce7c55dbffeb7ef1007e25e97bcb716",
            "avutil-60.dll": "6f172b5d10224fcc3f729c8baa6fd36a97bb58042bb3b1417078d77d2da59b87",
            "swresample-6.dll": "72e2721672c11fd37d983b05cc2370f612784e4e3218362a0cb4315d08c917fb",
            "swscale-9.dll": "3d07972cada6ba38c492e92b0f6c025a6835607fe00cdf83afc604c2fbdfe550",
        },
    },
    "candidate": {
        "version": CANDIDATE_VERSION,
        "distribution_version": "9.0.1-full_build-www.gyan.dev",
        "configuration_sha256": "ad62c89377019485be134b59fea8b5c7cf2c88f46a1521d5526081105ad068f0",
        "ffmpeg_sha256": "cf6b46df53d3672e86af7662358bbd2b21c90cc78c133f3f81f46e63acc387b3",
        "ffprobe_sha256": "0c49675a5f3098b881b1508368deb22e23d22fe56fa7a197df1b98c4e5ea79bf",
        "runtime_libraries": {
            "avcodec-63.dll": "f958e8ae31ce50b58e228c354411e406cd46c0021a6d250e90cf007fe65740d3",
            "avdevice-63.dll": "cc2de2187efd18aed52d3021d90934337bfe0e8ec60d988797b87ae7664f5ee0",
            "avfilter-12.dll": "8f8e2d63f6658450169d7fe2f5696fa9b01df3c1d3820cf706e142ba80758924",
            "avformat-63.dll": "8c0615789d41737051cf082351d4b9c869dd2f0abac4b792ff838041638752e5",
            "avutil-61.dll": "e289456490e190e0d74aa34980aeaa68903a6656248e2e7ef830e17acd80eb49",
            "swresample-7.dll": "d240955beb927ff2fb46cc4f80f83db20a10a9b9032f4092a10f492836fb0213",
            "swscale-10.dll": "89df1925fc718639cb13e849bc940dca114a10e97d93f8fdfd6c14369941a964",
        },
    },
}

FORBIDDEN_HARDWARE_TERMS = (
    "amf",
    "cuda",
    "cuvid",
    "d3d11va",
    "dxva2",
    "mediacodec",
    "nvenc",
    "qsv",
    "vaapi",
    "videotoolbox",
    "vulkan",
)

STREAM_KEYS = (
    "codec_name",
    "profile",
    "codec_type",
    "codec_tag_string",
    "width",
    "height",
    "coded_width",
    "coded_height",
    "closed_captions",
    "film_grain",
    "has_b_frames",
    "sample_aspect_ratio",
    "display_aspect_ratio",
    "pix_fmt",
    "level",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "chroma_location",
    "field_order",
    "refs",
    "is_avc",
    "nal_length_size",
    "r_frame_rate",
    "avg_frame_rate",
    "time_base",
    "start_pts",
    "duration_ts",
    "nb_frames",
    "extradata_size",
)

FRAME_KEYS = (
    "media_type",
    "stream_index",
    "key_frame",
    "pts",
    "duration",
    "best_effort_timestamp",
    "width",
    "height",
    "pix_fmt",
    "sample_aspect_ratio",
    "pict_type",
    "interlaced_frame",
    "top_field_first",
    "repeat_pict",
    "color_range",
    "color_space",
    "color_primaries",
    "color_transfer",
    "chroma_location",
    "crop_top",
    "crop_bottom",
    "crop_left",
    "crop_right",
)


class DifferentialError(RuntimeError):
    """A bounded verification condition was not met."""


@dataclass(frozen=True)
class Toolchain:
    label: str
    version: str
    root: Path
    ffmpeg: Path
    ffprobe: Path
    ffmpeg_identity: dict[str, str]
    ffprobe_identity: dict[str, str]
    runtime_libraries: dict[str, str]


@dataclass(frozen=True)
class Case:
    name: str
    filename: str
    required_encoder: str
    required_decoder: str
    decode_pix_fmt: str
    expected_frames: int
    generation_steps: tuple[tuple[str, ...], ...]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DifferentialError(f"cannot hash {path.name}: {error}") from error
    return digest.hexdigest()


def require_path_free(value: object, forbidden: tuple[Path, ...]) -> None:
    needles = {
        spelling.casefold() if os.name == "nt" else spelling
        for path in forbidden
        for spelling in (str(path), path.as_posix())
        if spelling
    }

    def visit(item: object) -> None:
        if isinstance(item, str):
            candidate = item.casefold() if os.name == "nt" else item
            if any(needle in candidate for needle in needles):
                raise DifferentialError("receipt unexpectedly contains a local path")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)


def executable(root: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    binary = root / "bin" / f"{name}{suffix}"
    if not binary.is_file() or binary.is_symlink():
        raise DifferentialError(
            f"SDK root must contain a regular, non-symlink bin/{name}{suffix}"
        )
    return binary


def expected_identities(manifest_path: Path | None) -> dict[str, dict[str, object]]:
    if manifest_path is None:
        if os.name != "nt":
            raise DifferentialError(
                "non-Windows verification requires --identity-manifest with exact binary/library hashes"
            )
        return WINDOWS_GYAN_IDENTITIES
    try:
        resolved = manifest_path.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise DifferentialError("identity manifest must be a regular, non-symlink file")
        if resolved.stat().st_size > 64 * 1024:
            raise DifferentialError("identity manifest exceeds 65536 bytes")
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DifferentialError(f"cannot read identity manifest: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise DifferentialError("identity manifest must use schema_version 1")
    if set(document) != {"schema_version", "baseline", "candidate"}:
        raise DifferentialError("identity manifest has missing or unexpected fields")
    result = {"baseline": document["baseline"], "candidate": document["candidate"]}
    if not all(isinstance(value, dict) for value in result.values()):
        raise DifferentialError("identity manifest SDK entries must be objects")
    return result  # type: ignore[return-value]


def validated_expected_identity(
    expected: dict[str, object], label: str, version: str
) -> dict[str, object]:
    required = {
        "version",
        "distribution_version",
        "configuration_sha256",
        "ffmpeg_sha256",
        "ffprobe_sha256",
        "runtime_libraries",
    }
    if set(expected) != required or expected.get("version") != version:
        raise DifferentialError(f"{label} identity manifest shape/version is invalid")
    for field in (
        "configuration_sha256",
        "ffmpeg_sha256",
        "ffprobe_sha256",
    ):
        value = expected.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise DifferentialError(f"{label} identity has invalid {field}")
    distribution = expected.get("distribution_version")
    if not isinstance(distribution, str) or not distribution.startswith(version):
        raise DifferentialError(f"{label} identity has invalid distribution_version")
    libraries = expected.get("runtime_libraries")
    if not isinstance(libraries, dict) or not libraries or len(libraries) > 32:
        raise DifferentialError(f"{label} runtime library inventory is absent or unbounded")
    for name, digest in libraries.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise DifferentialError(f"{label} runtime library identity is invalid")
    return expected


def validate_runtime_libraries(
    root: Path, label: str, expected: dict[str, str]
) -> dict[str, str]:
    if os.name == "nt":
        actual_names = {
            path.name
            for path in (root / "bin").glob("*.dll")
            if path.is_file()
        }
        if actual_names != set(expected):
            raise DifferentialError(
                f"{label} runtime DLL inventory mismatch: "
                f"missing={sorted(set(expected) - actual_names)}, "
                f"unexpected={sorted(actual_names - set(expected))}"
            )
    validated: dict[str, str] = {}
    for name, expected_hash in sorted(expected.items()):
        matches = [
            directory / name
            for directory in (root / "bin", root / "lib", root / "lib64")
            if (directory / name).exists()
        ]
        if len(matches) != 1:
            raise DifferentialError(
                f"{label} runtime library {name!r} must resolve exactly once"
            )
        library = matches[0]
        if not library.is_file() or library.is_symlink():
            raise DifferentialError(f"{label} runtime library {name!r} is not regular")
        actual_hash = sha256_file(library)
        if actual_hash != expected_hash:
            raise DifferentialError(f"{label} runtime library {name!r} hash mismatch")
        validated[name] = actual_hash
    return validated


def work_tree_size(work: Path) -> int:
    total = 0
    try:
        for path in work.rglob("*"):
            if path.is_symlink():
                raise DifferentialError("temporary work tree contains a symlink")
            if path.is_file():
                total += path.stat().st_size
                if total > MAX_TOTAL_WORK_BYTES:
                    return total
    except OSError as error:
        raise DifferentialError(f"cannot account temporary work tree: {error}") from error
    return total


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def bounded_command(
    args: list[str],
    work: Path,
    label: str,
    *,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
    output_cap: int = MAX_COMMAND_OUTPUT_BYTES,
    watched_paths: tuple[Path, ...] = (),
    watched_file_cap: int = MAX_FIXTURE_BYTES,
) -> tuple[bytes, bytes]:
    """Run without a shell while bounding time and both captured byte streams."""
    if timeout <= 0 or output_cap <= 0 or watched_file_cap <= 0:
        raise DifferentialError(f"{label}: command bounds must be positive")
    lowered = tuple(arg.lower() for arg in args[1:])
    for arg in lowered:
        if any(
            re.search(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)", arg)
            for term in FORBIDDEN_HARDWARE_TERMS
        ):
            raise DifferentialError(f"{label}: hardware term is forbidden in command arguments")

    if bounded_command.sequence >= MAX_COMMANDS:
        raise DifferentialError(f"{label}: command count exceeded {MAX_COMMANDS}")
    bounded_command.sequence += 1
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "FFREPORT": "",
            "NO_COLOR": "1",
        }
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    reader_errors: list[BaseException] = []

    def read_stream(stream: object, key: str) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)  # type: ignore[attr-defined]
                if not chunk:
                    return
                remaining = output_cap - len(buffers[key])
                if remaining > 0:
                    buffers[key].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    return
        except BaseException as error:  # surfaced on the controlling thread
            reader_errors.append(error)

    try:
        process = subprocess.Popen(
            args,
            cwd=work,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise DifferentialError(f"{label}: could not execute bounded command: {error}") from error
    assert process.stdout is not None and process.stderr is not None
    readers = [
        threading.Thread(target=read_stream, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=read_stream, args=(process.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    failure: str | None = None
    while process.poll() is None:
        if overflow.is_set():
            failure = f"command output exceeded {output_cap} bytes"
        elif time.monotonic() >= deadline:
            failure = f"exceeded {timeout:g}-second timeout"
        else:
            try:
                for path in watched_paths:
                    if path.is_symlink():
                        failure = "generation created a symlink"
                        break
                    if path.is_file() and path.stat().st_size > watched_file_cap:
                        failure = f"generated output exceeded {watched_file_cap} bytes"
                        break
                if failure is None and work_tree_size(work) > MAX_TOTAL_WORK_BYTES:
                    failure = f"temporary work tree exceeded {MAX_TOTAL_WORK_BYTES} bytes"
            except (OSError, DifferentialError) as error:
                failure = f"resource accounting failed: {error}"
        if failure is not None:
            terminate_process_tree(process)
            break
        time.sleep(0.01)
    if process.poll() is None:
        terminate_process_tree(process)
    return_code = process.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=5)
    process.stdout.close()
    process.stderr.close()
    if any(reader.is_alive() for reader in readers):
        raise DifferentialError(f"{label}: output reader did not terminate")
    if reader_errors:
        raise DifferentialError(f"{label}: output reader failed: {reader_errors[0]}")
    if failure is None and overflow.is_set():
        failure = f"command output exceeded {output_cap} bytes"
    if failure is None:
        try:
            for path in watched_paths:
                if path.is_symlink():
                    failure = "generation created a symlink"
                    break
                if path.is_file() and path.stat().st_size > watched_file_cap:
                    failure = f"generated output exceeded {watched_file_cap} bytes"
                    break
            if failure is None and work_tree_size(work) > MAX_TOTAL_WORK_BYTES:
                failure = f"temporary work tree exceeded {MAX_TOTAL_WORK_BYTES} bytes"
        except (OSError, DifferentialError) as error:
            failure = f"resource accounting failed: {error}"
    if failure is not None:
        raise DifferentialError(f"{label}: {failure}")
    stdout = bytes(buffers["stdout"])
    stderr = bytes(buffers["stderr"])
    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[-2000:]
        raise DifferentialError(f"{label}: exited {return_code}: {detail}")
    return stdout, stderr


bounded_command.sequence = 0


def command_identity(binary: Path, name: str, version: str, work: Path) -> dict[str, str]:
    stdout, _ = bounded_command([str(binary), "-version"], work, f"identify {name} {version}")
    lines = stdout.decode("utf-8", errors="strict").splitlines()
    expected = re.compile(rf"^{re.escape(name)} version {re.escape(version)}(?:[-+ ]|$)")
    if not lines or expected.match(lines[0]) is None:
        actual = lines[0] if lines else "<empty>"
        raise DifferentialError(f"expected exact {name} {version}, got {actual!r}")
    configuration = next((line for line in lines if line.startswith("configuration:")), "")
    if name == "ffmpeg" and not configuration:
        raise DifferentialError(f"{name} {version} omitted its build configuration")
    distribution = lines[0].split(" Copyright", 1)[0].removeprefix(f"{name} version ")
    return {
        "binary_sha256": sha256_file(binary),
        "configuration_sha256": sha256_bytes(configuration.encode("utf-8")),
        "distribution_version": distribution,
        "version_line": lines[0],
    }


def validate_command_identities(
    ffmpeg_identity: dict[str, str],
    ffprobe_identity: dict[str, str],
    expected: dict[str, object],
    label: str,
) -> None:
    if (
        ffmpeg_identity["distribution_version"]
        != ffprobe_identity["distribution_version"]
        or ffmpeg_identity["configuration_sha256"]
        != ffprobe_identity["configuration_sha256"]
    ):
        raise DifferentialError(
            f"{label} ffmpeg and ffprobe do not identify the same reviewed build"
        )
    for actual, field in (
        (ffmpeg_identity["binary_sha256"], "ffmpeg_sha256"),
        (ffprobe_identity["binary_sha256"], "ffprobe_sha256"),
        (ffmpeg_identity["configuration_sha256"], "configuration_sha256"),
        (ffmpeg_identity["distribution_version"], "distribution_version"),
    ):
        if actual != expected[field]:
            raise DifferentialError(f"{label} exact {field} identity mismatch")


def load_toolchain(
    root_arg: Path,
    label: str,
    version: str,
    work: Path,
    expected: dict[str, object],
) -> Toolchain:
    try:
        root = root_arg.resolve(strict=True)
    except OSError as error:
        raise DifferentialError(f"{label} SDK root does not resolve: {error}") from error
    if not root.is_dir():
        raise DifferentialError(f"{label} SDK root is not a directory")
    ffmpeg = executable(root, "ffmpeg")
    ffprobe = executable(root, "ffprobe")
    ffmpeg_identity = command_identity(ffmpeg, "ffmpeg", version, work)
    ffprobe_identity = command_identity(ffprobe, "ffprobe", version, work)
    expected = validated_expected_identity(expected, label, version)
    validate_command_identities(ffmpeg_identity, ffprobe_identity, expected, label)
    libraries = validate_runtime_libraries(
        root,
        label,
        expected["runtime_libraries"],  # type: ignore[arg-type]
    )
    return Toolchain(
        label=label,
        version=version,
        root=root,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        ffmpeg_identity=ffmpeg_identity,
        ffprobe_identity=ffprobe_identity,
        runtime_libraries=libraries,
    )


def listed_names(tool: Path, option: str, work: Path, label: str) -> set[str]:
    stdout, _ = bounded_command(
        [str(tool), "-hide_banner", option], work, f"enumerate {label} {option}"
    )
    names: set[str] = set()
    for line in stdout.decode("utf-8", errors="strict").splitlines():
        match = re.match(r"^\s*[A-Z.]{6}\s+(\S+)", line)
        if match:
            names.add(match.group(1))
    return names


def listed_simple_names(tool: Path, option: str, work: Path, label: str) -> set[str]:
    stdout, _ = bounded_command(
        [str(tool), "-hide_banner", option], work, f"enumerate {label} {option}"
    )
    names: set[str] = set()
    for line in stdout.decode("utf-8", errors="strict").splitlines():
        item = line.strip()
        if item and re.fullmatch(r"[a-zA-Z0-9_]+", item):
            names.add(item)
    return names


def fixture_cases() -> tuple[Case, ...]:
    common = ("-hide_banner", "-nostdin", "-loglevel", "error", "-y")
    return (
        Case(
            name="h264_long_gop_bframes_sar_rotation_crop_bt709_limited_8bit",
            filename="h264-crop-rotate.mp4",
            required_encoder="libx264",
            required_decoder="h264",
            decode_pix_fmt="yuv420p",
            expected_frames=16,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=12",
                    "-frames:v",
                    "16",
                    "-vf",
                    "setsar=4/3,format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-g",
                    "16",
                    "-bf",
                    "3",
                    "-x264-params",
                    "threads=1:keyint=16:min-keyint=16:bframes=3:scenecut=0:open-gop=0:colorprim=bt709:transfer=bt709:colormatrix=bt709:range=limited",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "-flags:v",
                    "+bitexact",
                    "$STAGE0",
                ),
                common
                + (
                    "-hwaccel",
                    "none",
                    "-i",
                    "$STAGE0",
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    "-bsf:v",
                    "h264_metadata=crop_right=2:crop_bottom=2",
                    "-map_metadata",
                    "-1",
                    "$STAGE1",
                ),
                common
                + (
                    "-hwaccel",
                    "none",
                    "-display_rotation:v:0",
                    "90",
                    "-i",
                    "$STAGE1",
                    "-map",
                    "0:v:0",
                    "-c",
                    "copy",
                    "-map_metadata",
                    "-1",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="hevc_bt2020_limited_10bit",
            filename="hevc-main10.mp4",
            required_encoder="libx265",
            required_decoder="hevc",
            decode_pix_fmt="yuv420p10le",
            expected_frames=8,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=8",
                    "-frames:v",
                    "8",
                    "-vf",
                    "format=yuv420p10le,setparams=range=limited:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
                    "-c:v",
                    "libx265",
                    "-preset",
                    "ultrafast",
                    "-g",
                    "8",
                    "-bf",
                    "2",
                    "-x265-params",
                    "log-level=error:pools=1:frame-threads=1:wpp=0:keyint=8:min-keyint=8:bframes=2:scenecut=0:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt2020nc",
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    "smpte2084",
                    "-tag:v",
                    "hvc1",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="vp9_bt601_full_8bit",
            filename="vp9-full.webm",
            required_encoder="libvpx-vp9",
            required_decoder="vp9",
            decode_pix_fmt="yuv420p",
            expected_frames=8,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=8",
                    "-frames:v",
                    "8",
                    "-vf",
                    "setrange=full,format=yuv420p",
                    "-c:v",
                    "libvpx-vp9",
                    "-deadline",
                    "good",
                    "-cpu-used",
                    "8",
                    "-row-mt",
                    "0",
                    "-tile-columns",
                    "0",
                    "-frame-parallel",
                    "0",
                    "-auto-alt-ref",
                    "0",
                    "-g",
                    "8",
                    "-color_range",
                    "pc",
                    "-colorspace",
                    "smpte170m",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="av1_bt2020_limited_10bit",
            filename="av1-main10.mkv",
            required_encoder="libaom-av1",
            required_decoder="av1",
            decode_pix_fmt="yuv420p10le",
            expected_frames=6,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=6",
                    "-frames:v",
                    "6",
                    "-vf",
                    "format=yuv420p10le,setparams=range=limited:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc",
                    "-c:v",
                    "libaom-av1",
                    "-usage",
                    "realtime",
                    "-cpu-used",
                    "8",
                    "-row-mt",
                    "0",
                    "-tiles",
                    "1x1",
                    "-g",
                    "6",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt2020nc",
                    "-color_primaries",
                    "bt2020",
                    "-color_trc",
                    "smpte2084",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="ffv1_vfr_nonzero_start",
            filename="ffv1-vfr.mkv",
            required_encoder="ffv1",
            required_decoder="ffv1",
            decode_pix_fmt="yuv420p",
            expected_frames=6,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=12",
                    "-frames:v",
                    "6",
                    "-vf",
                    "select=eq(mod(n\\,3)\\,0)+eq(mod(n\\,5)\\,0),setpts=PTS+5/TB,format=yuv420p",
                    "-fps_mode",
                    "passthrough",
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-coder",
                    "1",
                    "-context",
                    "1",
                    "-g",
                    "1",
                    "-threads",
                    "1",
                    "-slicecrc",
                    "1",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="mpeg2_interlaced_bt709_limited_8bit",
            filename="mpeg2-interlaced.mkv",
            required_encoder="mpeg2video",
            required_decoder="mpeg2video",
            decode_pix_fmt="yuv420p",
            expected_frames=6,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=12",
                    "-frames:v",
                    "6",
                    "-vf",
                    "format=yuv420p,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709,tinterlace=mode=interleave_top",
                    "-c:v",
                    "mpeg2video",
                    "-threads",
                    "1",
                    "-flags:v",
                    "+ilme+ildct+bitexact",
                    "-top",
                    "1",
                    "-g",
                    "6",
                    "-bf",
                    "0",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="qtrle_varying_alpha_8bit",
            filename="qtrle-alpha.mov",
            required_encoder="qtrle",
            required_decoder="qtrle",
            decode_pix_fmt="rgba",
            expected_frames=4,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "nullsrc=size=64x48:rate=8,format=rgba,geq=r=X/W*255:g=Y/H*255:b=(X+Y)/(W+H)*255:a=(X+N*8)/W*255",
                    "-frames:v",
                    "4",
                    "-c:v",
                    "qtrle",
                    "-pix_fmt",
                    "argb",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
        Case(
            name="mpeg4_simple_profile_8bit",
            filename="mpeg4-simple.avi",
            required_encoder="mpeg4",
            required_decoder="mpeg4",
            decode_pix_fmt="yuv420p",
            expected_frames=8,
            generation_steps=(
                common
                + (
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=64x48:rate=10",
                    "-frames:v",
                    "8",
                    "-vf",
                    "format=yuv420p",
                    "-c:v",
                    "mpeg4",
                    "-profile:v",
                    "0",
                    "-g",
                    "8",
                    "-bf",
                    "0",
                    "-q:v",
                    "4",
                    "-map_metadata",
                    "-1",
                    "-fflags",
                    "+bitexact",
                    "-flags:v",
                    "+bitexact",
                    "$FIXTURE",
                ),
            ),
        ),
    )


def require_capabilities(baseline: Toolchain, candidate: Toolchain, cases: tuple[Case, ...], work: Path) -> None:
    baseline_encoders = listed_names(
        baseline.ffmpeg, "-encoders", work, "baseline encoders"
    )
    baseline_decoders = listed_names(
        baseline.ffmpeg, "-decoders", work, "baseline decoders"
    )
    candidate_decoders = listed_names(
        candidate.ffmpeg, "-decoders", work, "candidate decoders"
    )
    missing_encoders = sorted(
        {case.required_encoder for case in cases} - baseline_encoders
    )
    missing_baseline_decoders = sorted(
        {case.required_decoder for case in cases} - baseline_decoders
    )
    missing_candidate_decoders = sorted(
        {case.required_decoder for case in cases} - candidate_decoders
    )
    baseline_bsfs = listed_simple_names(
        baseline.ffmpeg, "-bsfs", work, "baseline bitstream filters"
    )
    if missing_encoders or missing_baseline_decoders or missing_candidate_decoders:
        raise DifferentialError(
            "required codec capability is missing: "
            f"baseline_encoders={missing_encoders}, "
            f"baseline_decoders={missing_baseline_decoders}, "
            f"candidate_decoders={missing_candidate_decoders}"
        )
    if "h264_metadata" not in baseline_bsfs:
        raise DifferentialError("baseline lacks required h264_metadata crop bitstream filter")


def materialize_args(
    recipe: tuple[str, ...], fixture: Path, stages: tuple[Path, Path]
) -> list[str]:
    replacements = {
        "$FIXTURE": str(fixture),
        "$STAGE0": str(stages[0]),
        "$STAGE1": str(stages[1]),
    }
    return [replacements.get(arg, arg) for arg in recipe]


def generate_fixture(case: Case, baseline: Toolchain, work: Path) -> tuple[Path, str]:
    fixture = work / case.filename
    stages = (work / f"{case.name}-stage-0.mp4", work / f"{case.name}-stage-1.mp4")
    recipe_hashes: list[str] = []
    for index, recipe in enumerate(case.generation_steps):
        recipe_hashes.append(sha256_bytes(canonical_json(recipe)))
        args = [str(baseline.ffmpeg), *materialize_args(recipe, fixture, stages)]
        bounded_command(
            args,
            work,
            f"generate {case.name} step {index + 1}",
            watched_paths=(*stages, fixture),
        )
        for generated in (*stages, fixture):
            if not generated.exists():
                continue
            if not generated.is_file() or generated.is_symlink():
                raise DifferentialError(
                    f"{case.name}: generation created a non-regular output"
                )
            if generated.stat().st_size > MAX_FIXTURE_BYTES:
                raise DifferentialError(
                    f"{case.name}: generated output exceeds {MAX_FIXTURE_BYTES} bytes"
                )
    if not fixture.is_file() or fixture.is_symlink():
        raise DifferentialError(f"{case.name}: generation did not create a regular fixture")
    size = fixture.stat().st_size
    if size <= 0 or size > MAX_FIXTURE_BYTES:
        raise DifferentialError(
            f"{case.name}: fixture size {size} is outside 1..{MAX_FIXTURE_BYTES}"
        )
    return fixture, sha256_bytes(canonical_json(recipe_hashes))


def normalized_side_data(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise DifferentialError("ffprobe side-data entry is not an object")
        side_type = item.get("side_data_type")
        if side_type == "Display Matrix":
            normalized.append(
                {"side_data_type": side_type, "rotation": item.get("rotation")}
            )
    return normalized


def normalize_probe(document: object) -> dict[str, object]:
    if not isinstance(document, dict):
        raise DifferentialError("ffprobe document is not an object")
    streams = document.get("streams")
    frames = document.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        raise DifferentialError("fixture must contain exactly one probed video stream")
    if not isinstance(frames, list) or not (1 <= len(frames) <= MAX_FRAMES_PER_CASE):
        raise DifferentialError(
            f"fixture frame count must be within 1..{MAX_FRAMES_PER_CASE}"
        )
    stream = {key: streams[0][key] for key in STREAM_KEYS if key in streams[0]}
    stream["display_side_data"] = normalized_side_data(streams[0].get("side_data_list"))
    normalized_frames: list[dict[str, object]] = []
    for raw in frames:
        if not isinstance(raw, dict):
            raise DifferentialError("ffprobe frame entry is not an object")
        frame = {key: raw[key] for key in FRAME_KEYS if key in raw}
        frame["display_side_data"] = normalized_side_data(raw.get("side_data_list"))
        normalized_frames.append(frame)
    return {"stream": stream, "frames": normalized_frames}


def probe_fixture(toolchain: Toolchain, fixture: Path, work: Path, case: Case) -> dict[str, object]:
    stdout, _ = bounded_command(
        [
            str(toolchain.ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_frames",
            "-threads",
            "1",
            "-flags:v",
            "+bitexact",
            "-of",
            "json",
            str(fixture),
        ],
        work,
        f"probe {case.name} with {toolchain.label}",
    )
    try:
        return normalize_probe(json.loads(stdout.decode("utf-8", errors="strict")))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DifferentialError(f"{case.name}: invalid ffprobe JSON: {error}") from error


def parse_framehash(stdout: bytes, case_name: str) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for line in stdout.decode("utf-8", errors="strict").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or not re.fullmatch(r"[0-9a-f]{64}", fields[5]):
            raise DifferentialError(f"{case_name}: malformed framehash record")
        try:
            record = {
                "stream": int(fields[0]),
                "dts": int(fields[1]),
                "pts": int(fields[2]),
                "duration": int(fields[3]),
                "size": int(fields[4]),
                "sha256": fields[5],
            }
        except ValueError as error:
            raise DifferentialError(f"{case_name}: non-integer framehash timing") from error
        if record["stream"] != 0 or record["size"] <= 0:
            raise DifferentialError(f"{case_name}: invalid decoded framehash record")
        frames.append(record)
    if not (1 <= len(frames) <= MAX_FRAMES_PER_CASE):
        raise DifferentialError(
            f"{case_name}: decoded frame count is outside 1..{MAX_FRAMES_PER_CASE}"
        )
    return frames


def decoded_frames(toolchain: Toolchain, fixture: Path, work: Path, case: Case) -> list[dict[str, object]]:
    stdout, _ = bounded_command(
        [
            str(toolchain.ffmpeg),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-threads:v",
            "1",
            "-flags:v",
            "+bitexact",
            "-noautorotate",
            "-i",
            str(fixture),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-threads",
            "1",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            case.decode_pix_fmt,
            "-fflags",
            "+bitexact",
            "-flags:v",
            "+bitexact",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ],
        work,
        f"decode {case.name} with {toolchain.label}",
    )
    return parse_framehash(stdout, case.name)


def require_equal(label: str, baseline: object, candidate: object) -> None:
    if baseline != candidate:
        raise DifferentialError(
            f"{label} mismatch: baseline_sha256={sha256_bytes(canonical_json(baseline))}, "
            f"candidate_sha256={sha256_bytes(canonical_json(candidate))}"
        )


def require_case_semantics(case: Case, probe: dict[str, object], frames: list[dict[str, object]]) -> None:
    stream = probe["stream"]
    metadata_frames = probe["frames"]
    assert isinstance(stream, dict)
    assert isinstance(metadata_frames, list)
    if len(frames) != case.expected_frames or len(metadata_frames) != case.expected_frames:
        raise DifferentialError(
            f"{case.name}: expected exactly {case.expected_frames} decoded/probed frames"
        )
    pts = [int(frame["pts"]) for frame in frames]
    if pts != sorted(pts) or len(set(pts)) != len(pts):
        raise DifferentialError(f"{case.name}: decoded PTS are not unique and ascending")

    def expect(field: str, value: object) -> None:
        if stream.get(field) != value:
            raise DifferentialError(
                f"{case.name}: expected stream {field}={value!r}, got {stream.get(field)!r}"
            )

    if case.name.startswith("h264_"):
        expect("codec_name", "h264")
        expect("pix_fmt", "yuv420p")
        expect("width", 62)
        expect("height", 46)
        expect("coded_width", 64)
        expect("coded_height", 48)
        expect("sample_aspect_ratio", "4:3")
        expect("color_range", "tv")
        expect("color_space", "bt709")
        expect("color_primaries", "bt709")
        expect("color_transfer", "bt709")
        expect("chroma_location", "left")
        expect("field_order", "progressive")
        if int(stream.get("has_b_frames", 0)) < 1:
            raise DifferentialError(f"{case.name}: B-frame signaling is absent")
        if not any(frame.get("pict_type") == "B" for frame in metadata_frames):
            raise DifferentialError(f"{case.name}: decoded B frames are absent")
        if {item.get("rotation") for item in stream["display_side_data"]} != {90}:
            raise DifferentialError(f"{case.name}: exact 90-degree display matrix is absent")
    elif case.name.startswith("hevc_"):
        expect("codec_name", "hevc")
        expect("pix_fmt", "yuv420p10le")
        expect("profile", "Main 10")
        expect("color_range", "tv")
        expect("color_space", "bt2020nc")
        expect("color_primaries", "bt2020")
        expect("color_transfer", "smpte2084")
        expect("chroma_location", "left")
        expect("field_order", "progressive")
    elif case.name.startswith("vp9_"):
        expect("codec_name", "vp9")
        expect("pix_fmt", "yuv420p")
        expect("color_range", "pc")
        expect("color_space", "smpte170m")
        expect("profile", "Profile 0")
        expect("field_order", "progressive")
    elif case.name.startswith("av1_"):
        expect("codec_name", "av1")
        expect("pix_fmt", "yuv420p10le")
        expect("color_range", "tv")
        expect("color_space", "bt2020nc")
        expect("color_primaries", "bt2020")
        expect("color_transfer", "smpte2084")
        expect("profile", "Main")
        expect("field_order", "progressive")
    elif case.name.startswith("ffv1_vfr"):
        expect("codec_name", "ffv1")
        expect("pix_fmt", "yuv420p")
        expect("color_range", "tv")
        expect("field_order", "progressive")
        if int(stream.get("start_pts", 0)) <= 0:
            raise DifferentialError(f"{case.name}: nonzero start PTS is absent")
        deltas = {right - left for left, right in zip(pts, pts[1:])}
        if len(deltas) < 2:
            raise DifferentialError(f"{case.name}: variable frame intervals are absent")
    elif case.name.startswith("mpeg2_interlaced"):
        expect("codec_name", "mpeg2video")
        expect("profile", "Main")
        expect("pix_fmt", "yuv420p")
        expect("color_range", "tv")
        expect("color_space", "bt709")
        expect("color_primaries", "bt709")
        expect("color_transfer", "bt709")
        expect("chroma_location", "left")
        expect("field_order", "tb")
        if not any(frame.get("interlaced_frame") == 1 for frame in metadata_frames):
            raise DifferentialError(f"{case.name}: interlaced decoded frames are absent")
    elif case.name.startswith("qtrle_"):
        expect("codec_name", "qtrle")
        expect("pix_fmt", "argb")
        expect("field_order", "progressive")
        if len({frame["sha256"] for frame in frames}) < 2:
            raise DifferentialError(f"{case.name}: varying alpha fixture frames are not distinct")
    elif case.name.startswith("mpeg4_"):
        expect("codec_name", "mpeg4")
        expect("profile", "Simple Profile")
        expect("pix_fmt", "yuv420p")
        expect("chroma_location", "left")
        if any(frame.get("pict_type") == "B" for frame in metadata_frames):
            raise DifferentialError(f"{case.name}: Simple Profile fixture unexpectedly has B frames")


def verify_case(case: Case, baseline: Toolchain, candidate: Toolchain, work: Path) -> dict[str, object]:
    fixture, generation_recipe_sha256 = generate_fixture(case, baseline, work)
    baseline_probe = probe_fixture(baseline, fixture, work, case)
    candidate_probe = probe_fixture(candidate, fixture, work, case)
    baseline_frames = decoded_frames(baseline, fixture, work, case)
    candidate_frames = decoded_frames(candidate, fixture, work, case)
    require_equal(f"{case.name} normalized probe", baseline_probe, candidate_probe)
    require_equal(f"{case.name} decoded frames", baseline_frames, candidate_frames)
    require_case_semantics(case, baseline_probe, baseline_frames)
    return {
        "case": case.name,
        "codec": case.required_decoder,
        "decode_pix_fmt": case.decode_pix_fmt,
        "decoded_frame_count": len(baseline_frames),
        "decoded_frames_sha256": sha256_bytes(canonical_json(baseline_frames)),
        "fixture_sha256": sha256_file(fixture),
        "fixture_size": fixture.stat().st_size,
        "generation_recipe_sha256": generation_recipe_sha256,
        "normalized_probe_sha256": sha256_bytes(canonical_json(baseline_probe)),
    }


def toolchain_receipt(toolchain: Toolchain) -> dict[str, object]:
    return {
        "expected_version": toolchain.version,
        "ffmpeg": toolchain.ffmpeg_identity,
        "ffprobe": toolchain.ffprobe_identity,
        "runtime_libraries": toolchain.runtime_libraries,
    }


RECEIPT_FIELDS = {
    "schema_version",
    "receipt_kind",
    "verified",
    "identity_policy",
    "hardware_acceleration",
    "limits",
    "resource_accounting",
    "application_motion_scope",
    "baseline",
    "candidate",
    "matrix",
}

CASE_RECEIPT_FIELDS = {
    "case",
    "codec",
    "decode_pix_fmt",
    "decoded_frame_count",
    "decoded_frames_sha256",
    "fixture_sha256",
    "fixture_size",
    "generation_recipe_sha256",
    "normalized_probe_sha256",
}


def validate_receipt_schema(receipt: dict[str, object], cases: tuple[Case, ...]) -> None:
    if (
        set(receipt) != RECEIPT_FIELDS
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_kind")
        != "collide_o_scope_ffmpeg_software_differential"
        or receipt.get("verified") is not True
        or receipt.get("hardware_acceleration") != "disabled"
    ):
        raise DifferentialError("differential receipt schema/header is invalid")
    matrix = receipt.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != len(cases):
        raise DifferentialError("differential receipt matrix is incomplete")
    if [row.get("case") for row in matrix if isinstance(row, dict)] != [
        case.name for case in cases
    ]:
        raise DifferentialError("differential receipt case order/inventory is invalid")
    for row, case in zip(matrix, cases):
        if (
            not isinstance(row, dict)
            or set(row) != CASE_RECEIPT_FIELDS
            or row.get("decoded_frame_count") != case.expected_frames
        ):
            raise DifferentialError(f"{case.name}: differential receipt row is invalid")
        for field in (
            "decoded_frames_sha256",
            "fixture_sha256",
            "generation_recipe_sha256",
            "normalized_probe_sha256",
        ):
            if not isinstance(row.get(field), str) or re.fullmatch(
                r"[0-9a-f]{64}", row[field]
            ) is None:
                raise DifferentialError(f"{case.name}: receipt hash is invalid")


def verify(
    baseline_root: Path,
    candidate_root: Path,
    identity_manifest: Path | None = None,
) -> dict[str, object]:
    bounded_command.sequence = 0
    self_test()
    identities = expected_identities(identity_manifest)
    with tempfile.TemporaryDirectory(prefix="cos-ffmpeg-software-differential-") as raw_temp:
        work = Path(raw_temp)
        baseline = load_toolchain(
            baseline_root,
            "baseline",
            BASELINE_VERSION,
            work,
            identities["baseline"],
        )
        candidate = load_toolchain(
            candidate_root,
            "candidate",
            CANDIDATE_VERSION,
            work,
            identities["candidate"],
        )
        if baseline.ffmpeg == candidate.ffmpeg or baseline.ffprobe == candidate.ffprobe:
            raise DifferentialError("baseline and candidate SDK binaries must be distinct")
        cases = fixture_cases()
        if len(cases) != 8 or len({case.name for case in cases}) != 8:
            raise DifferentialError("exact eight-case differential matrix is required")
        require_capabilities(baseline, candidate, cases, work)
        results = [verify_case(case, baseline, candidate, work) for case in cases]
        total_size = sum(int(result["fixture_size"]) for result in results)
        if total_size > MAX_TOTAL_FIXTURE_BYTES:
            raise DifferentialError(
                f"total fixture size {total_size} exceeds {MAX_TOTAL_FIXTURE_BYTES} bytes"
            )
        work_tree_size(work)
        receipt: dict[str, object] = {
            "schema_version": 1,
            "receipt_kind": "collide_o_scope_ffmpeg_software_differential",
            "verified": True,
            "identity_policy": "exact_binary_configuration_and_runtime_library_sha256",
            "hardware_acceleration": "disabled",
            "limits": {
                "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
                "max_command_output_bytes": MAX_COMMAND_OUTPUT_BYTES,
                "max_fixture_bytes": MAX_FIXTURE_BYTES,
                "max_frames_per_case": MAX_FRAMES_PER_CASE,
                "max_total_fixture_bytes": MAX_TOTAL_FIXTURE_BYTES,
                "max_total_work_bytes": MAX_TOTAL_WORK_BYTES,
                "max_commands": MAX_COMMANDS,
            },
            "resource_accounting": {
                "retained_command_output_bytes": 0,
                "total_final_fixture_bytes": total_size,
                "stage_files_included_in_total_work_cap": True,
                "command_output_streamed_without_log_files": True,
            },
            "application_motion_scope": {
                "status": "not_evaluated_by_this_cli_differential",
                "reason": "motion-vector side data and Codec Mosh policy are application-level behaviors",
                "required_companion_gate": "cargo test --locked --all-targets --all-features codec_mosh::tests -- --ignored --nocapture",
                "required_companion_test_count": 5,
            },
            "baseline": toolchain_receipt(baseline),
            "candidate": toolchain_receipt(candidate),
            "matrix": results,
        }
        validate_receipt_schema(receipt, cases)
        require_path_free(
            receipt,
            (
                work,
                baseline.ffmpeg.parents[1],
                candidate.ffmpeg.parents[1],
            ),
        )
        return receipt


def self_test() -> None:
    if len(fixture_cases()) != 8 or len({case.name for case in fixture_cases()}) != 8:
        raise DifferentialError("self-test: fixture matrix shape changed unexpectedly")
    with tempfile.TemporaryDirectory(prefix="cos-ffmpeg-differential-self-test-") as raw_temp:
        root = Path(raw_temp)
        try:
            executable(root, "ffmpeg")
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: missing SDK binary was accepted")

        payload = {"path": "$FIXTURE", "stable": True}
        first = canonical_json(payload)
        second = canonical_json({"stable": True, "path": "$FIXTURE"})
        if first != second or raw_temp.encode("utf-8") in first:
            raise DifferentialError("self-test: canonical path-free JSON is unstable")
        try:
            require_path_free({"leak": str(root)}, (root,))
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: local path leakage was accepted")

        parsed = parse_framehash(
            b"#format: frame checksums\n0, 0, 0, 1, 4, " + b"0" * 64 + b"\n",
            "self-test",
        )
        if len(parsed) != 1 or parsed[0]["sha256"] != "0" * 64:
            raise DifferentialError("self-test: valid framehash was not parsed exactly")
        try:
            parse_framehash(b"0, 0, 0, 1, 4, not-a-hash\n", "self-test")
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: malformed framehash was accepted")

        try:
            require_equal("self-test mutation", {"value": 1}, {"value": 2})
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: differential mutation was accepted")

        try:
            bounded_command(
                ["does-not-exist", "-c:v", "h264_nvenc"],
                root,
                "self-test hardware rejection",
            )
        except DifferentialError as error:
            if "hardware term" not in str(error):
                raise DifferentialError(
                    "self-test: hardware command failed for the wrong reason"
                ) from error
        else:
            raise DifferentialError("self-test: hardware command was accepted")

        output_work = root / "output-cap"
        output_work.mkdir()
        try:
            bounded_command(
                [
                    sys.executable,
                    "-c",
                    "import sys;sys.stdout.buffer.write(b'x'*4096)",
                ],
                output_work,
                "self-test output cap",
                output_cap=1024,
            )
        except DifferentialError as error:
            if "output exceeded" not in str(error):
                raise DifferentialError(
                    "self-test: output cap failed for the wrong reason"
                ) from error
        else:
            raise DifferentialError("self-test: output cap was not enforced")

        timeout_work = root / "timeout"
        timeout_work.mkdir()
        try:
            bounded_command(
                [sys.executable, "-c", "import time;time.sleep(2)"],
                timeout_work,
                "self-test timeout",
                timeout=0.05,
            )
        except DifferentialError as error:
            if "exceeded 0.05-second timeout" not in str(error):
                raise DifferentialError(
                    "self-test: timeout failed for the wrong reason"
                ) from error
        else:
            raise DifferentialError("self-test: timeout was not enforced")

        fixture_work = root / "fixture-cap"
        fixture_work.mkdir()
        oversized = fixture_work / "oversized.bin"
        try:
            bounded_command(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,sys,time;pathlib.Path(sys.argv[1]).write_bytes(b'x'*4096);time.sleep(2)",
                    str(oversized),
                ],
                fixture_work,
                "self-test fixture cap",
                timeout=1,
                watched_paths=(oversized,),
                watched_file_cap=1024,
            )
        except DifferentialError as error:
            if "generated output exceeded" not in str(error):
                raise DifferentialError(
                    "self-test: fixture cap failed for the wrong reason"
                ) from error
        else:
            raise DifferentialError("self-test: fixture cap was not enforced")

        expected = validated_expected_identity(
            WINDOWS_GYAN_IDENTITIES["baseline"],
            "self-test",
            BASELINE_VERSION,
        )
        ffmpeg_identity = {
            "binary_sha256": expected["ffmpeg_sha256"],
            "configuration_sha256": expected["configuration_sha256"],
            "distribution_version": expected["distribution_version"],
            "version_line": "self-test",
        }
        ffprobe_identity = {
            "binary_sha256": expected["ffprobe_sha256"],
            "configuration_sha256": expected["configuration_sha256"],
            "distribution_version": expected["distribution_version"],
            "version_line": "self-test",
        }
        validate_command_identities(
            ffmpeg_identity,  # type: ignore[arg-type]
            ffprobe_identity,  # type: ignore[arg-type]
            expected,
            "self-test",
        )
        ffmpeg_identity["binary_sha256"] = "0" * 64
        try:
            validate_command_identities(
                ffmpeg_identity,  # type: ignore[arg-type]
                ffprobe_identity,  # type: ignore[arg-type]
                expected,
                "self-test mutation",
            )
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: tampered executable identity was accepted")

        library_root = root / "library-identity"
        (library_root / "bin").mkdir(parents=True)
        library_name = "self-test.dll" if os.name == "nt" else "libself-test.so"
        library = library_root / "bin" / library_name
        library.write_bytes(b"reviewed library bytes")
        library_hash = sha256_file(library)
        validate_runtime_libraries(
            library_root, "self-test", {library_name: library_hash}
        )
        library.write_bytes(b"tampered library bytes")
        try:
            validate_runtime_libraries(
                library_root, "self-test mutation", {library_name: library_hash}
            )
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: tampered runtime library was accepted")

        mpeg4 = next(case for case in fixture_cases() if case.name.startswith("mpeg4_"))
        semantic_frames = [
            {
                "stream": 0,
                "dts": index,
                "pts": index,
                "duration": 1,
                "size": 4,
                "sha256": f"{index:064x}",
            }
            for index in range(mpeg4.expected_frames)
        ]
        semantic_probe: dict[str, object] = {
            "stream": {
                "codec_name": "mpeg4",
                "profile": "Simple Profile",
                "pix_fmt": "yuv420p",
                "chroma_location": "left",
            },
            "frames": [
                {"pict_type": "I" if index == 0 else "P"}
                for index in range(mpeg4.expected_frames)
            ],
        }
        require_case_semantics(mpeg4, semantic_probe, semantic_frames)
        semantic_probe["stream"]["pix_fmt"] = "yuv444p"  # type: ignore[index]
        try:
            require_case_semantics(mpeg4, semantic_probe, semantic_frames)
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: semantic mutation was accepted")

        cases = fixture_cases()
        mock_matrix = [
            {
                "case": case.name,
                "codec": case.required_decoder,
                "decode_pix_fmt": case.decode_pix_fmt,
                "decoded_frame_count": case.expected_frames,
                "decoded_frames_sha256": "0" * 64,
                "fixture_sha256": "0" * 64,
                "fixture_size": 1,
                "generation_recipe_sha256": "0" * 64,
                "normalized_probe_sha256": "0" * 64,
            }
            for case in cases
        ]
        mock_receipt: dict[str, object] = {
            "schema_version": 1,
            "receipt_kind": "collide_o_scope_ffmpeg_software_differential",
            "verified": True,
            "identity_policy": "self-test",
            "hardware_acceleration": "disabled",
            "limits": {},
            "resource_accounting": {},
            "application_motion_scope": {},
            "baseline": {},
            "candidate": {},
            "matrix": mock_matrix,
        }
        validate_receipt_schema(mock_receipt, cases)
        del mock_matrix[0]["fixture_sha256"]
        try:
            validate_receipt_schema(mock_receipt, cases)
        except DifferentialError:
            pass
        else:
            raise DifferentialError("self-test: malformed receipt schema was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-sdk",
        type=Path,
        help=f"explicit shared SDK root containing FFmpeg {BASELINE_VERSION}",
    )
    parser.add_argument(
        "--candidate-sdk",
        type=Path,
        help=f"explicit shared SDK root containing FFmpeg {CANDIDATE_VERSION}",
    )
    parser.add_argument(
        "--identity-manifest",
        type=Path,
        help="exact cross-platform binary/configuration/runtime-library SHA-256 manifest; Windows defaults to the reviewed Gyan identities",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise bounded argument, path, receipt, and mismatch rejection",
    )
    args = parser.parse_args()
    try:
        if args.self_test:
            if (
                args.baseline_sdk is not None
                or args.candidate_sdk is not None
                or args.identity_manifest is not None
            ):
                raise DifferentialError("--self-test does not accept SDK arguments")
            self_test()
            print("FFmpeg software differential self-test passed")
            return 0
        if args.baseline_sdk is None or args.candidate_sdk is None:
            raise DifferentialError("both --baseline-sdk and --candidate-sdk are required")
        receipt = verify(
            args.baseline_sdk,
            args.candidate_sdk,
            args.identity_manifest,
        )
    except DifferentialError as error:
        print(f"FFmpeg software differential failed: {error}", file=sys.stderr)
        return 1
    print(canonical_json(receipt).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
