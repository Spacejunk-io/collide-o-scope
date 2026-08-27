#!/usr/bin/env python3
"""Static fail-closed checks for release workflow trust pins and gates."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REVIEWED_REPRODUCIBLE_BUILD_SHA256 = (
    "a2059888b1d9f29cac96491f840c71afba27fd0192ff72412a56cd5c29416e2c"
)
REVIEWED_PINNED_LLVM_STEP_SHA256 = (
    "350df6cb05ff8a9ec6eb61963784afafb454a586314569c27961374880f61484"
)
REVIEWED_RELEASE_SOURCE_RESOLUTION_STEP_SHA256 = (
    "a4b91f799b9d142edb172702358d5c6000cd4b2432bde901db7809020bd77d62"
)
REVIEWED_PRISTINE_CHECKOUT_STEP_SHA256 = (
    "42a4f40c3a227af08e26c3d8b39fe2ae1eb43cd0632303dde434f2bd0ad996a4"
)
REVIEWED_PRETAG_FINAL_ABSENCE_STEP_SHA256 = (
    "166d8802ecf67c23a3116a65eddfce7e3df06bf832442693391da73362641e51"
)
REVIEWED_UNEQUAL_PREPARE_STEP_SHA256 = (
    "aac71777897fef9c04aa898322c62ad24fd70638091ee595fd88c5cafe3a3efc"
)
REVIEWED_UNEQUAL_BUILD_STEP_SHA256 = (
    "d4ab78f545f3eb0a23a93f303b7fc8b492c6d832904d0e305344eef997e335f7"
)
REVIEWED_UNEQUAL_CONTIGUOUS_REGION_SHA256 = (
    "73ae6a31bbed06c823e157ecfd3571c5e289f34274e37b598569672311f05ad1"
)
REVIEWED_REPRODUCIBLE_SBOM_STEP_SHA256 = (
    "58576a9f07f6a9a6c3fcb8c7fdbb49a13a6eb15993550cb0a9679a21d2372833"
)
REVIEWED_PACKAGE_ASSEMBLY_STEP_SHA256 = (
    "f327da1c8f9029e9b224bffe95b1d5713b9cee875428c56733ff04f15e468cbb"
)
REVIEWED_SBOM_POLICY_SHA256 = (
    "775741aeb0e652e52a83c59364a27f543ef58ca51b486ed48e127d76f57769e9"
)
REVIEWED_RELEASE_VERIFIER_SHA256 = (
    "904cfa1fac528996daca88ac17799cf0a1f06652cb621a26f6f477f4207801a4"
)

CHECKOUT_ACTION = (
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1"
)
CACHE_ACTION = (
    "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9 # v6.1.0"
)
COSIGN_INSTALLER_ACTION = (
    "sigstore/cosign-installer@6f9f17788090df1f26f669e9d70d6ae9567deba6 # v4.1.2"
)
ATTEST_BUILD_PROVENANCE_ACTION = (
    "actions/attest-build-provenance@977bb373ede98d70efdf65b84cb5f73e068dcc2a # v3.0.0"
)
UPLOAD_ARTIFACT_ACTION = (
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2"
)
EXPECTED_WORKFLOW_ACTIONS = {
    "adversarial.yml": Counter({CHECKOUT_ACTION: 2, CACHE_ACTION: 2}),
    "ci.yml": Counter({CHECKOUT_ACTION: 2, CACHE_ACTION: 4}),
    "release-trust.yml": Counter(
        {
            CHECKOUT_ACTION: 5,
            CACHE_ACTION: 1,
            COSIGN_INSTALLER_ACTION: 2,
            ATTEST_BUILD_PROVENANCE_ACTION: 2,
            UPLOAD_ARTIFACT_ACTION: 1,
        }
    ),
}
REVIEWED_WORKFLOW_SHA256 = {
    "adversarial.yml": (
        "4f8143f8316943894c0e545ba70e54fec4d4b8e227a4e429bd3f7b8c30028590"
    ),
    "ci.yml": "0848e77ff0bb959611c6a85f7e69ecca5ddf5464a3130942ad43ff7b2855a2b9",
    "release-trust.yml": (
        "7da4f2fb04d1107e8306d8d815866234c2022337d0326296ca5003433512212b"
    ),
}
PINNED_WORKFLOW_ACTION = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40} # v[0-9]+\.[0-9]+\.[0-9]+"
)
COSIGN_RELEASE = "v3.1.3"
COSIGN_GIT_COMMIT = "11926fa5bbbbde47e88fc006b625a17769b743b2"
COSIGN_WINDOWS_AMD64_SHA256 = (
    "9fe59be0eca1271873ce019061335eb1ac419b7059202e797828467ddabe33be"
)
SIGSTORE_BUNDLE_MEDIA_TYPE = "application/vnd.dev.sigstore.bundle.v0.3+json"
COSIGN_CERTIFICATE_ISSUER_ARGUMENT = (
    "--certificate-oidc-issuer https://token.actions.githubusercontent.com"
)
COSIGN_CERTIFICATE_IDENTITY_ARGUMENT = (
    '--certificate-identity "https://github.com/${{ github.repository }}/'
    '.github/workflows/release-trust.yml@${{ github.ref }}"'
)
NATIVE_EXIT_GUARD = "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"
REVIEWED_COSIGN_SIGNING_STEP_SHA256 = {
    "Keyless-sign the checksum manifest": (
        "e072709fd7b936f7292f6a36f7edaded72c019ab0a2e3ce6af62bf4bbe52b98a"
    ),
    "Keyless-sign the frozen final-release receipt": (
        "d98a46ebddb7db0334b3278921e59453fb7ae03d162d19c77b880d047b6a201a"
    ),
}
COSIGN_INSTALLER_STEP = f"""      - name: Install pinned cosign
        uses: {COSIGN_INSTALLER_ACTION}
        with:
          cosign-release: {COSIGN_RELEASE}"""
COSIGN_IDENTITY_STEP = f"""      - name: Verify pinned cosign identity
        shell: pwsh
        run: |
          $ErrorActionPreference = "Stop"
          $cosign = Get-Command cosign -CommandType Application -ErrorAction Stop
          $actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $cosign.Source).Hash.ToLowerInvariant()
          $expectedSha256 = "{COSIGN_WINDOWS_AMD64_SHA256}"
          if ($actualSha256 -cne $expectedSha256) {{
            throw "Installed cosign SHA-256 did not match the pinned Windows amd64 artifact"
          }}
          $versionJson = & $cosign.Source version --json
          {NATIVE_EXIT_GUARD}
          $version = $versionJson | ConvertFrom-Json
          if (
            $version.gitVersion -cne "{COSIGN_RELEASE}" -or
            $version.gitCommit -cne "{COSIGN_GIT_COMMIT}" -or
            $version.gitTreeState -cne "clean" -or
            $version.platform -cne "windows/amd64"
          ) {{
            throw "Installed cosign version identity did not match the pinned release"
          }}"""


def fail(message: str) -> None:
    raise ValueError(message)


def workflow_action_values(workflow: str) -> list[str]:
    values = []
    in_named_step = False
    block_scalar_indent: int | None = None
    block_scalar_header = re.compile(
        r"^(?P<indent> *)(?:-\s+)?"
        r'''(?:[A-Za-z0-9_.-]+|"(?:\\.|[^"])*"|'(?:''|[^'])*'):\s*'''
        r"[|>](?:[1-9][+-]?|[+-][1-9]?)?\s*(?:#.*)?$"
    )
    named_step = re.compile(r"^      - name:\s*[^\r\n]+?\s*$")
    action_like = re.compile(
        r'''^ *(?:-\s+)?(?:uses|"uses"|'uses'):\s*'''
        r"(?P<value>[^\r\n]+?)\s*$"
    )
    canonical_action = re.compile(r"^        uses:\s*(?P<value>[^\r\n]+?)\s*$")
    for line in workflow.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if block_scalar_indent is not None:
            if not stripped or stripped.startswith("#"):
                continue
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None
        if named_step.fullmatch(line) is not None:
            in_named_step = True
            continue
        if stripped and not stripped.startswith("#") and indent <= 6:
            in_named_step = False
        match = action_like.fullmatch(line)
        if match is not None:
            canonical = canonical_action.fullmatch(line)
            if canonical is None or not in_named_step:
                fail(
                    "workflow action uses must be an unquoted eight-space key "
                    "inside a canonical named step"
                )
            values.append(canonical.group("value"))
            continue
        header = block_scalar_header.fullmatch(line)
        if header is not None:
            block_scalar_indent = len(header.group("indent"))
    return values


def validate_pinned_workflow_actions(workflows: dict[str, str]) -> None:
    if set(workflows) != set(EXPECTED_WORKFLOW_ACTIONS):
        fail("reviewed workflow inventory differs from the exact expected set")
    for name, expected in EXPECTED_WORKFLOW_ACTIONS.items():
        if hashlib.sha256(workflows[name].encode()).hexdigest() != (
            REVIEWED_WORKFLOW_SHA256[name]
        ):
            fail(f"{name} differs from its reviewed complete document")
        values = workflow_action_values(workflows[name])
        if any(PINNED_WORKFLOW_ACTION.fullmatch(value) is None for value in values):
            fail(
                f"every {name} action must use a full lowercase commit SHA and "
                "its exact reviewed version comment"
            )
        if Counter(values) != expected:
            fail(f"{name} action pins or occurrence counts differ from review")


def self_test_pinned_workflow_actions(workflows: dict[str, str]) -> None:
    validate_pinned_workflow_actions(workflows)

    def changed(name: str, text: str) -> dict[str, str]:
        mutation = dict(workflows)
        mutation[name] = text
        return mutation

    adversarial = workflows["adversarial.yml"]
    ci = workflows["ci.yml"]
    checkout_line = f"        uses: {CHECKOUT_ACTION}"
    cache_line = f"        uses: {CACHE_ACTION}"
    checkout_step = f"""      - name: Check out source
{checkout_line}"""
    run_block_spoof = f"""      - name: Check out source
        run: |
          uses: {CHECKOUT_ACTION}"""
    quoted_run_block_spoof = f'''      - name: Check out source
        "run": |
          uses: {CHECKOUT_ACTION}'''
    env_spoof = f'''      - name: Check out source
        env:
          uses: {CHECKOUT_ACTION}
        run: echo "$env:uses"'''
    quoted_action = f'''      - name: Check out source
        "uses": {CHECKOUT_ACTION}'''
    escaped_action = f'''      - name: Escaped unknown action
        "u\\u0073es": actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v6.0.0'''
    inline_action = (
        "      - { name: Inline unknown action, uses: "
        "actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d } # v6.0.0"
    )
    hostile = [
        changed("adversarial.yml", adversarial.replace(checkout_line + "\n", "", 1)),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, run_block_spoof, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, quoted_run_block_spoof, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, env_spoof, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, quoted_action, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, checkout_step + "\n" + escaped_action, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(checkout_step, checkout_step + "\n" + inline_action, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(cache_line, cache_line + "\n" + cache_line, 1),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace(
                "3d3c42e5aac5ba805825da76410c181273ba90b1",
                "3d3c42e5aac5ba805825da76410c181273ba90b0",
                1,
            ),
        ),
        changed(
            "adversarial.yml",
            adversarial.replace("# v7.0.1", "# v7.0.0", 1),
        ),
        changed(
            "adversarial.yml",
            adversarial
            + "\n        uses: actions/setup-python@82c7e631bb3cdc910f68e0081d67478d79c6982d # v6.0.0\n",
        ),
    ]
    moved = dict(workflows)
    moved["adversarial.yml"] = adversarial.replace(checkout_line + "\n", "", 1)
    moved["ci.yml"] = ci.replace(checkout_line, checkout_line + "\n" + checkout_line, 1)
    hostile.append(moved)
    for mutation in hostile:
        if mutation == workflows:
            fail("workflow-action self-test fixture is absent")
        try:
            validate_pinned_workflow_actions(mutation)
        except ValueError:
            pass
        else:
            fail("workflow-action hostile mutation escaped the policy gate")


def workflow_named_step_blocks(workflow: str, name: str) -> list[str]:
    matches = list(re.finditer(r"(?m)^      - name: ([^\r\n]+)\s*$", workflow))
    blocks = []
    for index, match in enumerate(matches):
        if match.group(1) != name:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(workflow)
        blocks.append(workflow[match.start() : end].rstrip())
    return blocks


def cosign_command_blocks(workflow: str) -> list[tuple[str, str, bool]]:
    lines = workflow.splitlines()
    commands = []
    index = 0
    while index < len(lines):
        match = re.match(
            r"^cosign (sign|verify)-blob(?:\s|$)", lines[index].strip()
        )
        if match is None:
            index += 1
            continue
        kind = match.group(1)
        command_lines = [lines[index].strip()]
        end = index
        while command_lines[-1].endswith("`") and end + 1 < len(lines):
            end += 1
            command_lines.append(lines[end].strip())
        following = end + 1
        while following < len(lines) and not lines[following].strip():
            following += 1
        guarded = (
            following < len(lines) and lines[following].strip() == NATIVE_EXIT_GUARD
        )
        commands.append((kind, "\n".join(command_lines), guarded))
        index = end + 1
    return commands


def cosign_command_tokens(command: str) -> list[str]:
    def without_unquoted_comment(line: str) -> str:
        result = []
        quote: str | None = None
        index = 0
        while index < len(line):
            character = line[index]
            if character == "`" and quote != "'" and index + 1 < len(line):
                result.extend((character, line[index + 1]))
                index += 2
                continue
            if quote == "'":
                if character == "'" and index + 1 < len(line) and line[index + 1] == "'":
                    result.extend((character, line[index + 1]))
                    index += 2
                    continue
                if character == "'":
                    quote = None
            elif quote == '"':
                if character == '"':
                    quote = None
            elif character in ("'", '"'):
                quote = character
            elif character == "#":
                break
            result.append(character)
            index += 1
        return "".join(result)

    command = "\n".join(
        without_unquoted_comment(line) for line in command.splitlines()
    )
    command = re.sub(r"`\s*\n\s*", " ", command)
    return re.findall(r'''"(?:`.|[^"])*"|'(?:''|[^'])*'|\S+''', command)


def cosign_option_operand(tokens: list[str], option: str) -> str | None:
    positions = [index for index, token in enumerate(tokens) if token == option]
    if len(positions) != 1:
        return None
    position = positions[0]
    if position + 1 >= len(tokens) or tokens[position + 1].startswith("--"):
        return None
    return tokens[position + 1]


def validate_cosign_trust_policy(release: str) -> None:
    forbidden = (
        "--new-bundle-format=false",
        "--insecure-ignore-tlog",
        "--insecure-ignore-sct",
        "--certificate-identity-regexp",
        "--certificate-oidc-issuer-regexp",
    )
    if any(value in release for value in forbidden):
        fail("Cosign trust policy contains a forbidden legacy or insecure option")

    installers = workflow_named_step_blocks(release, "Install pinned cosign")
    identity_proofs = workflow_named_step_blocks(
        release, "Verify pinned cosign identity"
    )
    if (
        installers != [COSIGN_INSTALLER_STEP, COSIGN_INSTALLER_STEP]
        or identity_proofs != [COSIGN_IDENTITY_STEP, COSIGN_IDENTITY_STEP]
        or release.count(COSIGN_INSTALLER_STEP + "\n\n" + COSIGN_IDENTITY_STEP) != 2
    ):
        fail("both Cosign installations must be followed by the exact binary proof")

    commands = cosign_command_blocks(release)
    sign_commands = [command for command in commands if command[0] == "sign"]
    verify_commands = [command for command in commands if command[0] == "verify"]
    if len(sign_commands) != 2 or len(verify_commands) != 5:
        fail("release trust must contain exactly two sign-blob and five verify-blob calls")
    issuer_tokens = cosign_command_tokens(COSIGN_CERTIFICATE_ISSUER_ARGUMENT)
    identity_tokens = cosign_command_tokens(COSIGN_CERTIFICATE_IDENTITY_ARGUMENT)
    if len(issuer_tokens) != 2 or len(identity_tokens) != 2:
        fail("internal Cosign identity policy is malformed")
    for kind, command, guarded in commands:
        tokens = cosign_command_tokens(command)
        if cosign_option_operand(tokens, "--bundle") is None or not guarded:
            fail(f"every Cosign {kind}-blob call must use one bundle and fail immediately")
        if kind == "sign":
            if tokens.count("--yes") != 1:
                fail("every Cosign sign-blob call must remain explicitly non-interactive")
            continue
        if (
            cosign_option_operand(tokens, issuer_tokens[0]) != issuer_tokens[1]
            or cosign_option_operand(tokens, identity_tokens[0])
            != identity_tokens[1]
        ):
            fail("every Cosign verification must bind the exact workflow identity and issuer")

    signing_steps = {
        "Keyless-sign the checksum manifest": (
            "$checksumBundle",
            "Checksum signature bundle mediaType did not match the pinned schema",
        ),
        "Keyless-sign the frozen final-release receipt": (
            "$bundleDocument",
            "Final receipt signature bundle mediaType did not match the pinned schema",
        ),
    }
    media_assertions = 0
    for step_name, (media_variable, failure_message) in signing_steps.items():
        blocks = workflow_named_step_blocks(release, step_name)
        if len(blocks) != 1:
            fail(f"Cosign signing step {step_name!r} is not unique")
        block = blocks[0]
        if (
            hashlib.sha256(block.encode()).hexdigest()
            != REVIEWED_COSIGN_SIGNING_STEP_SHA256[step_name]
        ):
            fail(f"Cosign signing step {step_name!r} differs from its reviewed body")
        block_commands = cosign_command_blocks(block)
        if [command[0] for command in block_commands] != ["sign", "verify"]:
            fail(f"{step_name} must sign, inspect the bundle, then verify")
        sign_command = block_commands[0][1]
        bundle_path = cosign_option_operand(
            cosign_command_tokens(sign_command), "--bundle"
        )
        if bundle_path is None:
            fail(f"{step_name} does not name its signed bundle")
        media_guard = (
            f"          {media_variable} = Get-Content -Raw -LiteralPath "
            f"{bundle_path} | ConvertFrom-Json\n"
            f"          if ({media_variable}.mediaType -cne "
            f'"{SIGSTORE_BUNDLE_MEDIA_TYPE}") {{\n'
            f'            throw "{failure_message}"\n'
            "          }"
        )
        if block.count(media_guard) != 1:
            fail(f"{step_name} must directly parse and reject a wrong bundle media type")
        verify_bundle_path = cosign_option_operand(
            cosign_command_tokens(block_commands[1][1]), "--bundle"
        )
        if verify_bundle_path != bundle_path:
            fail(f"{step_name} does not verify the exact emitted bundle")
        sign_position = re.search(r"(?m)^ *cosign sign-blob(?:\s|$)", block)
        verify_position = re.search(r"(?m)^ *cosign verify-blob(?:\s|$)", block)
        media_position = block.find(media_guard)
        if sign_position is None or verify_position is None:
            fail(f"{step_name} does not contain executable sign and verify commands")
        assignments = list(
            re.finditer(
                rf"(?mi)^ *{re.escape(media_variable)}\s*=",
                block[: verify_position.start()],
            )
        )
        if len(assignments) != 1 or assignments[0].start() != media_position:
            fail(f"{step_name} can overwrite its parsed bundle before verification")
        media_assertions += 1
        positions = (
            sign_position.start(),
            media_position,
            verify_position.start(),
        )
        if positions != tuple(sorted(positions)) or len(set(positions)) != 3:
            fail(f"{step_name} does not inspect its bundle between signing and verification")
    if media_assertions != 2 or release.count(SIGSTORE_BUNDLE_MEDIA_TYPE) != 2:
        fail("both and only both signing steps must pin the standardized bundle media type")


def self_test_cosign_trust_policy(release: str) -> None:
    validate_cosign_trust_policy(release)
    first_sign = "cosign sign-blob --yes --bundle"
    first_verify = "cosign verify-blob `"
    checksum_sign = (
        "          cosign sign-blob --yes --bundle "
        "release\\SHA256SUMS.sigstore.json release\\SHA256SUMS\n"
        f"          {NATIVE_EXIT_GUARD}"
    )
    checksum_media_type = f'''          $checksumBundle = Get-Content -Raw -LiteralPath release\\SHA256SUMS.sigstore.json | ConvertFrom-Json
          if ($checksumBundle.mediaType -cne "{SIGSTORE_BUNDLE_MEDIA_TYPE}") {{
            throw "Checksum signature bundle mediaType did not match the pinned schema"
          }}'''
    checksum_throw = (
        'throw "Checksum signature bundle mediaType did not match the pinned schema"'
    )

    def comment_first_command(text: str, prefix: str) -> str:
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip().startswith(prefix):
                continue
            while True:
                content = lines[index].rstrip("\r\n")
                newline = lines[index][len(content) :]
                indent = content[: len(content) - len(content.lstrip())]
                continued = content.rstrip().endswith("`")
                lines[index] = f"{indent}# {content.lstrip()}{newline}"
                if not continued:
                    return "".join(lines)
                index += 1
        fail("Cosign commented-command self-test fixture is absent")

    hostile = [
        release.replace(COSIGN_IDENTITY_STEP, "", 1),
        release.replace(COSIGN_WINDOWS_AMD64_SHA256, "0" * 64, 1),
        release.replace('gitVersion -cne "v3.1.3"', 'gitVersion -cne "v3.1.2"', 1),
        release.replace(COSIGN_GIT_COMMIT, "0" * 40, 1),
        release.replace('gitTreeState -cne "clean"', 'gitTreeState -cne "dirty"', 1),
        release.replace('platform -cne "windows/amd64"', 'platform -cne "linux/amd64"', 1),
        release.replace("cosign-release: v3.1.3", "cosign-release: v3.1.2", 1),
        release.replace(first_sign, "cosign sign-blob --yes", 1),
        release.replace(first_sign, "cosign sign-blob --yes=false --bundle", 1),
        release.replace(first_sign, "cosign sign-blob --yes --bundle-path", 1),
        release.replace(first_verify, "cosign version `", 1),
        comment_first_command(release, first_verify),
        release.replace(
            "--bundle release\\SHA256SUMS.sigstore.json `",
            "--signature release\\SHA256SUMS.sigstore.json `",
            1,
        ),
        release.replace(
            "--bundle release\\SHA256SUMS.sigstore.json `",
            "--bundle-path release\\SHA256SUMS.sigstore.json `",
            1,
        ),
        release.replace(
            "--bundle release\\SHA256SUMS.sigstore.json `",
            "# --bundle release\\SHA256SUMS.sigstore.json `",
            1,
        ),
        release.replace(
            "--bundle release\\SHA256SUMS.sigstore.json `",
            "--bundle release\\WRONG.sigstore.json `",
            1,
        ),
        release.replace(
            COSIGN_CERTIFICATE_ISSUER_ARGUMENT,
            "--certificate-oidc-issuer https://issuer.invalid",
            1,
        ),
        release.replace(
            COSIGN_CERTIFICATE_IDENTITY_ARGUMENT,
            '--certificate-identity "https://github.com/acme/other/.github/workflows/release-trust.yml@${{ github.ref }}"',
            1,
        ),
        release.replace(
            COSIGN_CERTIFICATE_IDENTITY_ARGUMENT,
            "# " + COSIGN_CERTIFICATE_IDENTITY_ARGUMENT,
            1,
        ),
        release.replace("--certificate-identity ", "--certificate-identity-regexp ", 1),
        release.replace("--certificate-oidc-issuer ", "--certificate-oidc-issuer-regexp ", 1),
        release.replace(SIGSTORE_BUNDLE_MEDIA_TYPE, "application/example+json", 1),
        release.replace(
            'throw "Checksum signature bundle mediaType did not match the pinned schema"',
            'Write-Warning "Checksum signature bundle mediaType did not match the pinned schema"',
            1,
        ),
        release.replace(
            'throw "Final receipt signature bundle mediaType did not match the pinned schema"',
            'Write-Warning "Final receipt signature bundle mediaType did not match the pinned schema"',
            1,
        ),
        release.replace(
            checksum_throw,
            f"if ($true) {{\n              {checksum_throw}\n            }}",
            1,
        ),
        release.replace(
            checksum_throw,
            f'''try {{
              {checksum_throw}
            }} catch {{
              Write-Warning "Swallowed media-type failure"
            }}''',
            1,
        ),
        release.replace(
            checksum_media_type,
            checksum_media_type
            + '\n          $checksumBundle = [pscustomobject]@{ mediaType = "trusted" }',
            1,
        ),
        release.replace(
            checksum_media_type,
            "          if ($false) {\n"
            + "  "
            + checksum_media_type.replace("\n", "\n  ")
            + "\n          }",
            1,
        ),
        release.replace(
            first_sign,
            "cosign sign-blob --new-bundle-format=false --yes --bundle",
            1,
        ),
        release.replace(
            first_verify, "cosign verify-blob --insecure-ignore-tlog `", 1
        ),
        release.replace(
            first_verify, "cosign verify-blob --insecure-ignore-sct `", 1
        ),
        release.replace(
            checksum_sign,
            checksum_sign + "\n" + checksum_sign,
            1,
        ),
        release.replace(
            checksum_sign + "\n" + checksum_media_type,
            checksum_media_type + "\n" + checksum_sign,
            1,
        ),
    ]
    guarded_sign = re.search(
        rf"(?m)^(?P<indent>\s*)cosign sign-blob[^\r\n]*\r?\n"
        rf"(?P=indent){re.escape(NATIVE_EXIT_GUARD)}\r?\n",
        release,
    )
    if guarded_sign is None:
        fail("Cosign native-guard self-test fixture is absent")
    hostile.append(
        release[: guarded_sign.start()]
        + guarded_sign.group(0).splitlines()[0]
        + "\n"
        + release[guarded_sign.end() :]
    )

    for mutation in hostile:
        if mutation == release:
            fail("Cosign trust self-test fixture is absent")
        try:
            validate_cosign_trust_policy(mutation)
        except ValueError:
            pass
        else:
            fail("Cosign trust hostile mutation escaped the policy gate")


def ci_job_section(workflow: str, job_id: str) -> tuple[int, int, str]:
    match = re.search(rf"(?m)^  {re.escape(job_id)}:\s*$", workflow)
    if match is None:
        fail(f"required CI job {job_id!r} is absent")
    following = re.search(r"(?m)^  [A-Za-z0-9_-]+:\s*$", workflow[match.end() :])
    end = len(workflow) if following is None else match.end() + following.start()
    return match.end(), end, workflow[match.end() : end]


def ci_named_steps(job: str) -> dict[str, dict[str, object]]:
    matches = list(re.finditer(r"(?m)^      - name: ([^\r\n]+)\s*$", job))
    steps: dict[str, dict[str, str | None]] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(job)
        block = job[match.start() : end]
        if name in steps:
            fail(f"CI step name {name!r} is duplicated in one job")
        shell_match = re.search(r"(?m)^        shell: ([^\s#]+)\s*$", block)
        condition_match = re.search(r"(?m)^        if: ([^\r\n]+)\s*$", block)
        block_run = re.search(r"(?m)^        run: \|\s*$", block)
        inline_run = re.search(r"(?m)^        run: ([^|\r\n].*)$", block)
        if (block_run is None) == (inline_run is None):
            run = None
        elif inline_run is not None:
            run = inline_run.group(1).rstrip() + "\n"
        else:
            assert block_run is not None
            lines = []
            run_source = block[block_run.end() :].lstrip("\r\n")
            for line in run_source.splitlines():
                if line and not line.startswith("          "):
                    break
                lines.append(line[10:] if line else "")
            run = "\n".join(lines).rstrip() + "\n"
        steps[name] = {
            "shell": shell_match.group(1) if shell_match else None,
            "condition": condition_match.group(1).strip() if condition_match else None,
            "run": run,
            "block": block,
            "keys": tuple(
                sorted(
                    re.findall(
                        r"(?m)^        ([A-Za-z][A-Za-z0-9_-]*):", block
                    )
                )
            ),
        }
    return steps


def expected_ci_gate_steps() -> dict[str, dict[str, tuple[str | None, str | None, str]]]:
    return {
        "test": {
            "Verify the vendored wgpu-hal archive and sole patch": (
                None,
                None,
                "python scripts/verify-vendored-wgpu-hal.py --self-test\n",
            ),
            "Check Rust formatting and JavaScript syntax": (
                "bash",
                None,
                "set -euo pipefail\n"
                "cargo fmt --all -- --check\n"
                "python scripts/verify-ffmpeg-software-differential.py --self-test\n"
                "node --check static/app.js\n"
                "node --check docs/ui-ux/wireframe.js\n",
            ),
            "Check generated capability registry": (
                None,
                None,
                "cargo run --locked --bin generate_capabilities -- --check\n",
            ),
            "Check, test, and lint on Unix": (
                "bash",
                "runner.os != 'Windows'",
                "set -euo pipefail\n"
                "cargo check --locked --all-targets --all-features\n"
                "cargo test --locked --all-targets --all-features\n"
                "cargo clippy --locked --all-targets --all-features -- -D warnings\n",
            ),
            "Check, test, and lint on Windows": (
                "pwsh",
                "runner.os == 'Windows'",
                'cmd /c "`"$env:VCVARS64`" >nul && cargo check --locked --all-targets --all-features"\n'
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
                'cmd /c "`"$env:VCVARS64`" >nul && cargo test --locked --all-targets --all-features"\n'
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
                'cmd /c "`"$env:VCVARS64`" >nul && cargo clippy --locked --all-targets --all-features -- -D warnings"\n'
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n",
            ),
        },
        "dependency-policy": {
            "Fetch locked dependencies and verify vendored source": (
                "bash",
                None,
                "set -euo pipefail\n"
                "cargo fetch --locked\n"
                "python scripts/verify-vendored-wgpu-hal.py --self-test\n",
            ),
            "Reject stale or unowned advisory exceptions": (
                "bash",
                None,
                "set -euo pipefail\n"
                "python scripts/check-dependency-policy.py\n"
                "python scripts/check-release-workflow.py\n"
                "python scripts/verify-release.py self-test\n",
            ),
        },
    }


def validate_ci_gate_bodies(workflow: str) -> None:
    if "continue-on-error:" in workflow or re.search(r"(?m)^\s+defaults:\s*$", workflow):
        fail("CI cannot tolerate failure in or around a required release gate")
    _, _, test_job = ci_job_section(workflow, "test")
    test_header = test_job.split("    steps:", 1)[0].strip()
    expected_test_header = """name: ${{ matrix.name }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 60
    strategy:
      fail-fast: false
      matrix:
        include:
          - name: Linux (Ubuntu 24.04)
            os: ubuntu-24.04
          - name: macOS 15
            os: macos-15
          - name: Windows (VS 2022)
            os: windows-2022"""
    _, _, dependency_job = ci_job_section(workflow, "dependency-policy")
    dependency_header = dependency_job.split("    steps:", 1)[0].strip()
    expected_dependency_header = """name: Dependency policy and supply-chain provenance
    runs-on: ubuntu-24.04
    timeout-minutes: 45"""
    if (
        test_header != expected_test_header
        or dependency_header != expected_dependency_header
    ):
        fail("required CI job names, runners, or matrix topology changed")
    for job_id, required in expected_ci_gate_steps().items():
        _, _, job = ci_job_section(workflow, job_id)
        steps = ci_named_steps(job)
        for name, (shell, condition, run) in required.items():
            step = steps.get(name)
            expected_keys = tuple(
                sorted(
                    {"run"}
                    | ({"shell"} if shell is not None else set())
                    | ({"if"} if condition is not None else set())
                )
            )
            if (
                step is None
                or step["shell"] != shell
                or step["condition"] != condition
                or step["run"] != run
                or step["keys"] != expected_keys
                or "continue-on-error:" in str(step["block"])
            ):
                fail(
                    f"required CI gate {job_id}/{name} no longer has its reviewed "
                    "fail-closed command body"
                )


def replace_ci_step_with_echo(workflow: str, job_id: str, step_name: str) -> str:
    job_start, _, job = ci_job_section(workflow, job_id)
    marker = f"      - name: {step_name}"
    relative_start = job.find(marker)
    if relative_start < 0:
        fail("CI body self-test could not find its required step")
    following = job.find("\n      - name: ", relative_start + len(marker))
    relative_end = len(job) if following < 0 else following + 1
    block = job[relative_start:relative_end]
    run_match = re.search(r"(?m)^        run: (?:\|.*|.*)$", block)
    if run_match is None:
        fail("CI body self-test could not find its required run body")
    mutated_block = block[: run_match.start()] + '        run: echo "test result: ok. 999 passed; 0 failed"\n'
    absolute_start = job_start + relative_start
    absolute_end = job_start + relative_end
    return workflow[:absolute_start] + mutated_block + workflow[absolute_end:]


def inject_ci_step_continue_on_error(workflow: str, job_id: str, step_name: str) -> str:
    job_start, _, job = ci_job_section(workflow, job_id)
    marker = f"      - name: {step_name}"
    relative_start = job.find(marker)
    if relative_start < 0:
        fail("CI tolerance self-test could not find its required step")
    insertion = relative_start + len(marker)
    absolute = job_start + insertion
    return workflow[:absolute] + "\n        continue-on-error: true" + workflow[absolute:]


def self_test_ci_gate_bodies(workflow: str) -> None:
    validate_ci_gate_bodies(workflow)
    structural_mutations = (
        workflow.replace("os: windows-2022", "os: ubuntu-24.04", 1),
        workflow.replace(
            "    timeout-minutes: 60",
            "    timeout-minutes: 60\n    defaults:\n      run:\n        shell: pwsh",
            1,
        ),
    )
    for mutation in structural_mutations:
        try:
            validate_ci_gate_bodies(mutation)
        except ValueError:
            pass
        else:
            fail("CI body self-test accepted changed job topology or defaults")
    for job_id, required in expected_ci_gate_steps().items():
        for step_name in required:
            mutation = replace_ci_step_with_echo(workflow, job_id, step_name)
            try:
                validate_ci_gate_bodies(mutation)
            except ValueError:
                pass
            else:
                fail(f"CI body self-test accepted echoed output for {job_id}/{step_name}")
            mutation = inject_ci_step_continue_on_error(workflow, job_id, step_name)
            try:
                validate_ci_gate_bodies(mutation)
            except ValueError:
                pass
            else:
                fail(f"CI body self-test accepted tolerated failure for {job_id}/{step_name}")


def validate_attestation_identity_policy(release: str) -> None:
    identity_assignment = (
        '$attestationIdentity = "https://github.com/${{ github.repository }}/'
        '.github/workflows/release-trust.yml@${{ github.ref }}"'
    )
    source_ref_assignment = '$attestationSourceRef = "refs/tags/$env:RELEASE_TAG"'
    attestation = re.compile(
        r"gh attestation verify (?P<subject>\$asset\.FullName|\$receipt|\$publishedReceipt) `\s+"
        r'--repo "\$\{\{ github\.repository \}\}" `\s+'
        r"--cert-identity \$attestationIdentity `\s+"
        r"--cert-oidc-issuer https://token\.actions\.githubusercontent\.com `\s+"
        r"--predicate-type https://slsa\.dev/provenance/v1 `\s+"
        r"--source-ref \$attestationSourceRef `\s+"
        r"--source-digest \$expected\s+"
        r"if \(\$LASTEXITCODE -ne 0\) \{ exit \$LASTEXITCODE \}"
    )
    matches = list(attestation.finditer(release))
    subjects = [match.group("subject") for match in matches]
    policy_document = [
        "schema_version = 2",
        "policy = [ordered]@{",
        'repository = "${{ github.repository }}"',
        "certificate_identity = $attestationIdentity",
        'certificate_oidc_issuer = "https://token.actions.githubusercontent.com"',
        'predicate_type = "https://slsa.dev/provenance/v1"',
        "source_ref = $attestationSourceRef",
        "source_digest = $expected",
    ]
    if (
        release.count(identity_assignment) != 2
        or release.count(source_ref_assignment) != 2
        or len(matches) != 4
        or subjects.count("$asset.FullName") != 2
        or subjects.count("$receipt") != 1
        or subjects.count("$publishedReceipt") != 1
        or any(release.count(value) < 1 for value in policy_document)
    ):
        fail(
            "GitHub attestations are not bound to the exact repository, signer, "
            "tag ref, source digest, issuer, and predicate"
        )


def self_test_attestation_identity_policy(release: str) -> None:
    validate_attestation_identity_policy(release)
    identity_assignment = (
        '$attestationIdentity = "https://github.com/${{ github.repository }}/'
        '.github/workflows/release-trust.yml@${{ github.ref }}"'
    )
    source_ref_assignment = '$attestationSourceRef = "refs/tags/$env:RELEASE_TAG"'
    attestation = re.compile(
        r"gh attestation verify (?:\$asset\.FullName|\$receipt|\$publishedReceipt) `\s+"
        r'--repo "\$\{\{ github\.repository \}\}" `\s+'
        r"--cert-identity \$attestationIdentity `\s+"
        r"--cert-oidc-issuer https://token\.actions\.githubusercontent\.com `\s+"
        r"--predicate-type https://slsa\.dev/provenance/v1 `\s+"
        r"--source-ref \$attestationSourceRef `\s+"
        r"--source-digest \$expected\s+"
        r"if \(\$LASTEXITCODE -ne 0\) \{ exit \$LASTEXITCODE \}"
    )
    hostile = [
        release.replace(identity_assignment, identity_assignment.replace("release-trust", "other"), 1),
        release.replace(source_ref_assignment, '$attestationSourceRef = "refs/heads/main"', 1),
        release.replace("certificate_identity = $attestationIdentity", "certificate_identity = other", 1),
    ]
    substitutions = (
        ("--repo", "--owner"),
        ("--cert-identity", "--cert-identity-regex"),
        ("--cert-oidc-issuer", "--custom-trusted-root"),
        ("--predicate-type", "--signer-workflow"),
        ("--source-ref", "--signer-repo"),
        ("--source-digest", "--signer-digest"),
    )
    for match in attestation.finditer(release):
        block = match.group(0)
        for original, replacement in substitutions:
            hostile_block = block.replace(original, replacement, 1)
            hostile.append(
                release[: match.start()] + hostile_block + release[match.end() :]
            )
    for mutation in hostile:
        try:
            validate_attestation_identity_policy(mutation)
        except ValueError:
            pass
        else:
            fail("attestation identity self-test accepted a weakened verification policy")


def validate_create_only_publication(release: str) -> None:
    section = release.split(
        "      - name: Create release and upload initial assets once", 1
    )
    if len(section) != 2:
        fail("create-only initial publication step is absent")
    publication = section[1].split("  redownload-verify:", 1)[0]
    ordered = [
        "verify-release.py release-absent",
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
        "$assets.Count -ne 10",
        'gh release create "$env:RELEASE_TAG" `',
        "--verify-tag `",
        "--draft `",
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
        "--json databaseId,tagName,isDraft",
        '"draft_id=$draftId" >> $env:GITHUB_OUTPUT',
        '"draft_tag=$($draft.tagName)" >> $env:GITHUB_OUTPUT',
        '"draft_state=$($draft.isDraft.ToString().ToLowerInvariant())" >> $env:GITHUB_OUTPUT',
        'gh release upload "$env:RELEASE_TAG" @assetPaths `',
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
        "[long]$uploadedDraft.databaseId -ne $draftId",
    ]
    cursor = 0
    for value in ordered:
        cursor = publication.find(value, cursor)
        if cursor < 0:
            fail("create-only initial publication sequence is incomplete")
        cursor += len(value)
    preflight = publication.find("verify-release.py release-absent")
    first_mutation = min(
        index
        for index in (
            publication.find("gh release create"),
            publication.find("gh release upload"),
            publication.find("gh release edit"),
        )
        if index >= 0
    )
    if (
        preflight < 0
        or preflight > first_mutation
        or "softprops/action-gh-release" in publication
        or "--clobber" in publication
        or "overwrite_files" in publication
        or publication.count("gh release create") != 1
        or publication.count("gh release upload") != 1
        or publication.count("--draft `") != 1
        or publication.count("--json databaseId,tagName,isDraft") != 2
        or '--target "${{ needs.verification-gate.outputs.commit }}"' not in publication
    ):
        fail("initial publication can mutate a preexisting release or overwrite assets")


def self_test_create_only_publication_policy() -> None:
    valid = """
      - name: Create release and upload initial assets once
        run: |
          python source-a\\scripts\\verify-release.py release-absent
          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
          if ($assets.Count -ne 10) { throw "wrong count" }
          gh release create "$env:RELEASE_TAG" `
            --target "${{ needs.verification-gate.outputs.commit }}" `
            --verify-tag `
            --draft `
            --generate-notes
          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
          gh release view --json databaseId,tagName,isDraft
          "draft_id=$draftId" >> $env:GITHUB_OUTPUT
          "draft_tag=$($draft.tagName)" >> $env:GITHUB_OUTPUT
          "draft_state=$($draft.isDraft.ToString().ToLowerInvariant())" >> $env:GITHUB_OUTPUT
          gh release upload "$env:RELEASE_TAG" @assetPaths `
            --repo repository
          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
          gh release view --json databaseId,tagName,isDraft
          if ([long]$uploadedDraft.databaseId -ne $draftId) { throw "changed" }
  redownload-verify:
"""
    validate_create_only_publication(valid)
    hostile = (
        valid.replace("verify-release.py release-absent", "verify-release.py verify"),
        valid.replace(
            "python source-a\\scripts\\verify-release.py release-absent",
            "gh release edit existing\n"
            "          python source-a\\scripts\\verify-release.py release-absent",
        ),
        valid.replace("--repo repository", "--repo repository --clobber"),
        valid.replace("            --draft `\n", ""),
        valid.replace(
            "python source-a\\scripts\\verify-release.py release-absent",
            "uses: softprops/action-gh-release@" + "a" * 40,
        ),
    )
    for mutation in hostile:
        try:
            validate_create_only_publication(mutation)
        except ValueError:
            pass
        else:
            fail("create-only publication self-test accepted a mutating rerun policy")


def validate_draft_publish_last(release: str) -> None:
    section = release.split("  redownload-verify:", 1)
    if len(section) != 2:
        fail("draft verification and final publication job is absent")
    redownload = section[1]
    final_step_parts = redownload.split(
        "      - name: Verify and publish the external final-release receipt", 1
    )
    if len(final_step_parts) != 2:
        fail("final receipt publication step is absent")
    final_step = final_step_parts[1]
    annotated_tag = re.compile(
        r"python scripts\\verify-release\.py annotated-tag `\s+"
        r'--tag "\$env:RELEASE_TAG" `\s+'
        r'--commit "\$\{\{ needs\.reproduce-sign-publish\.outputs\.commit \}\}" `\s+'
        r'--tag-object "\$\{\{ needs\.verification-gate\.outputs\.tag_object \}\}" `\s+'
        r"--remote origin\s+"
        r"if \(\$LASTEXITCODE -ne 0\) \{ exit \$LASTEXITCODE \}"
    )
    tag_checks = list(annotated_tag.finditer(final_step))
    ordered = [
        "[long]$draft.databaseId -ne [long]$env:EXPECTED_DRAFT_ID",
        '$draft.tagName -cne $env:EXPECTED_DRAFT_TAG',
        '$draft.isDraft -ne $true',
        'gh release download "$env:RELEASE_TAG" --repo',
        "[long]$beforeUpload.databaseId -ne $expectedDraftId",
        'gh release upload "$env:RELEASE_TAG" $receipt $bundle',
        '--notes-file "final-release\\release-summary.md"',
        '$view.isDraft -ne $true',
        '$view.body.TrimEnd() -cne $expectedBody',
        "--dir published-complete",
        "$publishedHash -cne $expectedHash",
        "gh attestation verify $publishedReceipt",
        "python scripts\\verify-release.py annotated-tag `",
        'gh release edit "$env:RELEASE_TAG" `',
        "--draft=false",
        "--json databaseId,tagName,url,body,assets,isDraft",
        '$publishedView.isDraft -ne $false',
        '$publishedView.url -cne $expectedUrl',
        '$publishedView.body.TrimEnd() -cne $expectedBody',
        "New-Item -ItemType Directory -Path published-final-state",
        "--dir published-final-state",
        "$finalHash -cne $draftHash",
        "Final published asset differs from verified draft bytes",
        "python scripts\\verify-release.py annotated-tag `",
    ]
    cursor = 0
    for value in ordered:
        cursor = redownload.find(value, cursor)
        if cursor < 0:
            fail("draft verification or publish-last persistence sequence is incomplete")
        cursor += len(value)
    final_edit = redownload.rfind('gh release edit "$env:RELEASE_TAG" `')
    final_upload = redownload.rfind("gh release upload")
    after_publish = redownload[final_edit:]
    final_step_edit = final_step.rfind('gh release edit "$env:RELEASE_TAG" `')
    final_hash_proof = final_step.find("Final published asset differs from verified draft bytes")
    if len(tag_checks) == 2:
        between_prepublish_check_and_edit = final_step[
            tag_checks[0].end() : final_step_edit
        ].strip()
        after_final_tag_check = final_step[tag_checks[1].end() :].strip()
    else:
        between_prepublish_check_and_edit = "missing"
        after_final_tag_check = "missing"
    if (
        redownload.count('gh release edit "$env:RELEASE_TAG" `') != 2
        or redownload.count("--draft=false") != 1
        or redownload.count('$view.isDraft -ne $true') != 1
        or redownload.count('$publishedView.isDraft -ne $false') != 1
        or final_edit < final_upload
        or redownload.find("gh release upload", final_edit) >= 0
        or redownload.find("gh release create", final_edit) >= 0
        or "gh release delete" in after_publish
        or re.search(r"gh api[^\n]*(?:--method|-X)\s+DELETE", after_publish, re.IGNORECASE)
        or "$view.url" in redownload[:final_edit]
        or redownload.count("published-final-state") < 3
        or redownload.count("$view.body.TrimEnd() -cne $expectedBody") != 1
        or redownload.count("$publishedView.url -cne $expectedUrl") != 1
        or redownload.count("$publishedView.body.TrimEnd() -cne $expectedBody") != 1
        or redownload.count("$publishedHash -cne $expectedHash") != 1
        or redownload.count("$finalHash -cne $draftHash") != 1
        or redownload.count("python scripts\\verify-release.py annotated-tag `") != 3
        or len(tag_checks) != 2
        or tag_checks[0].start() >= final_step_edit
        or between_prepublish_check_and_edit
        or final_hash_proof < final_step_edit
        or tag_checks[1].start() <= final_hash_proof
        or after_final_tag_check
    ):
        fail(
            "release can become public before all draft evidence is verified, "
            "or its tag can move across the public boundary"
        )


def self_test_draft_publish_last(release: str) -> None:
    validate_draft_publish_last(release)
    final_step_marker = "      - name: Verify and publish the external final-release receipt"
    prefix, final_step = release.split(final_step_marker, 1)
    tag_call = "python scripts\\verify-release.py annotated-tag `"
    first_tag = final_step.find(tag_call)
    second_tag = final_step.find(tag_call, first_tag + len(tag_call))
    if first_tag < 0 or second_tag < 0:
        fail("draft publication self-test fixture lacks both final tag checks")
    hostile = (
        release.replace("--draft=false", "--draft=true", 1),
        release.replace('$view.isDraft -ne $true', '$view.isDraft -ne $false', 1),
        release.replace(
            '$publishedView.isDraft -ne $false',
            '$publishedView.isDraft -ne $false\n          gh release upload after-publish',
            1,
        ),
        release.replace(
            '$publishedView.isDraft -ne $false',
            '$publishedView.isDraft -ne $false\n          gh release delete "$env:RELEASE_TAG"',
            1,
        ),
        release.replace(
            "$view.body.TrimEnd() -cne $expectedBody",
            "$view.body.TrimEnd() -ne $expectedBody",
            1,
        ),
        release.replace(
            "$publishedView.url -cne $expectedUrl",
            "$publishedView.url -ne $expectedUrl",
            1,
        ),
        release.replace(
            "$publishedView.body.TrimEnd() -cne $expectedBody",
            "$publishedView.body.TrimEnd() -ne $expectedBody",
            1,
        ),
        release.replace(
            "$publishedHash -cne $expectedHash",
            "$publishedHash -ne $expectedHash",
            1,
        ),
        release.replace(
            "$finalHash -cne $draftHash",
            "$finalHash -ne $draftHash",
            1,
        ),
        prefix
        + final_step_marker
        + final_step[:first_tag]
        + final_step[first_tag:].replace(tag_call, "python scripts\\verify-release.py verify `", 1),
        prefix
        + final_step_marker
        + final_step[:second_tag]
        + final_step[second_tag:].replace(tag_call, "python scripts\\verify-release.py verify `", 1),
    )
    for mutation in hostile:
        try:
            validate_draft_publish_last(mutation)
        except ValueError:
            pass
        else:
            fail("draft publication self-test accepted partial or post-publish mutation")


def canonical_reproducible_build_fragments() -> tuple[tuple[str, int], ...]:
    return (
        (r"$canonicalRoot = 'C:\cosrepro'", 1),
        (r"$canonicalMutexName = 'Local\CollideOScope.Repro.Stage.v1'", 1),
        ("canonical reproducible staging requires the fixed NTFS C: volume", 1),
        ("Assert-NoReparsePoints -Path 'C:\\'", 1),
        ("TargetDir must be absent", 1),
        ("[CollideReproducibleNativePaths]::GetLongPath((Resolve-Path -LiteralPath $requestedOutputParent).Path)", 1),
        ("function Assert-OnlyDefaultDataStream", 1),
        ("alternate data streams are not permitted", 1),
        ("ambient Cargo configuration or credentials are not permitted", 1),
        ("caller FFmpeg binaries must not be on PATH during the reproducible build", 1),
        ("Reproducible builds require an entirely clean source checkout", 1),
        ("SOURCE_DATE_EPOCH must equal the exact source commit timestamp", 1),
        ("tracked symlinks and gitlinks are not permitted", 1),
        ("& rustup run 1.98.0 rustc -vV", 1),
        ("& rustup run 1.98.0 cargo -Vv", 1),
        ("88d9e12ae178fab0fb5cc050a94da85685d449ea", 2),
        ("797e8a9bca276c1c9f9f738d2a20f484fa4eea9d", 2),
        ("80934e8f208a0cc2a87a6057f871d0f492461952b8672464749a6c3dff34109c", 1),
        ("51fed10c43c3d31c1fe5bfe76bac60150970961e9b9b23cf014dbfcb5398bbfc", 1),
        ("'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'", 1),
        ("'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'", 1),
        ("'DOCS_RS', 'RING_PREGENERATE_ASM', 'COLLIDE_BUILD_FFMPEG_BINARY'", 1),
        ("'SPOUT2_LIB_DIR', 'CC_FORCE_DISABLE'", 1),
        ("unreviewed ambient Cargo configuration is not permitted", 1),
        ("ambient CMake configuration is not permitted", 1),
        ("ambient AWS-LC routing is not permitted", 1),
        ("'GIT_DEFAULT_HASH', 'GIT_DEFAULT_REF_FORMAT', 'GIT_ALLOW_PROTOCOL'", 1),
        ("core.fsmonitor=false", 1),
        ("core.hooksPath=NUL", 1),
        ("$mutexHeld = $mutex.WaitOne(0)", 1),
        ("canonical reproducible staging root already exists and will not be auto-deleted", 1),
        ("[CollideReproducibleNativePaths]::CreateNewDirectory($canonicalRoot)", 1),
        ("[IO.FileMode]::CreateNew, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None", 1),
        ("archive --format=tar --output=$sourceArchive", 1),
        ("init --quiet --template=", 1),
        ("canonical source checkout disagrees with the exact clean input tree", 1),
        ("Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'registry\\cache')", 1),
        ("Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'registry\\index')", 1),
        ("Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'git\\db')", 1),
        ("Copy-VerifiedFile -Source (Join-Path $inputCargoSeed 'bin\\cargo-auditable.exe')", 1),
        ("expanded Cargo source/checkouts must be recreated offline", 1),
        ("foreach ($component in @('bin', 'include', 'lib'))", 1),
        ("Set-ProcessEnvironmentValue -Name 'CARGO_HOME' -Value $cargoHome", 1),
        ("Set-ProcessEnvironmentValue -Name 'CARGO_TARGET_DIR' -Value $target", 1),
        ("Set-ProcessEnvironmentValue -Name 'CARGO_NET_OFFLINE' -Value 'true'", 1),
        ("Set-ProcessEnvironmentValue -Name 'RUSTUP_TOOLCHAIN' -Value '1.98.0'", 1),
        ("Set-ProcessEnvironmentValue -Name $controlledArchiverName -Value $llvmAr", 1),
        ("Set-ProcessEnvironmentValue -Name $controlledAwsCmakeName -Value '0'", 1),
        ("Set-ProcessEnvironmentValue -Name $controlledAwsSystemName -Value '0'", 1),
        ("rustup run 1.98.0 cargo fetch --locked --offline", 1),
        (
            "$cargoGitCheckoutReview = [ordered]@{\n"
            "        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500' = '4b79500dfac64efcfb393eebc89f5c75565ee5ae'\n"
            "        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx' = '5aa788d5134f577c23eba158ded7592c4c471050'\n"
            "    }",
            1,
        ),
        (
            "$checkoutCommit = (git @gitSafetyArguments -C $checkoutPath rev-parse --verify 'HEAD^{commit}').Trim().ToLowerInvariant()",
            1,
        ),
        (
            "if ($LASTEXITCODE -ne 0 -or $checkoutCommit -cne $cargoGitCheckoutReview[$relativeCheckout]) {\n"
            "            throw \"Cargo Git checkout commit differs from Cargo.lock: $relativeCheckout\"\n"
            "        }",
            1,
        ),
        (
            "git @gitSafetyArguments -C $checkoutPath diff-index --quiet HEAD --\n"
            "        if ($LASTEXITCODE -ne 0) {\n"
            "            throw \"Cargo Git checkout differs from its reviewed commit: $relativeCheckout\"\n"
            "        }",
            1,
        ),
        (
            "$checkoutStatus = @(git @gitSafetyArguments -C $checkoutPath status --porcelain=v1 --untracked-files=all --ignore-submodules=dirty)",
            1,
        ),
        (
            "if ($LASTEXITCODE -ne 0 -or $checkoutStatus.Count -ne 1 -or $checkoutStatus[0] -cne '?? .cargo-ok') {",
            1,
        ),
        ("Cargo Git checkout marker is not the expected empty file", 1),
        ("Assert-OnlyDefaultDataStream -Path $cargoOk", 1),
        (
            "$relativeMetadataFiles = @(\n"
            "            \"$relativeCheckout/.git/index\",\n"
            "            \"$relativeCheckout/.git/logs/HEAD\",\n"
            "            \"$relativeCheckout/.git/logs/refs/heads/master\"\n"
            "        )",
            1,
        ),
        ("Assert-OnlyDefaultDataStream -Path $metadataPath", 1),
        ("$cargoCheckoutMetadataExclusions += $relativeMetadata", 1),
        (
            "$observedCargoCheckoutMetadata = @(\n"
            "        Get-ChildItem -LiteralPath (Join-Path $cargoHome 'git\\checkouts') -Force -Recurse -File |\n"
            "            ForEach-Object { $_.FullName.Substring($cargoHome.Length).TrimStart([char[]]@('\\', '/')).Replace('\\', '/') } |\n"
            "            Where-Object { $_ -match '/\\.git/index$' -or $_ -match '/\\.git/logs/' } |\n"
            "            Sort-Object\n"
            "    )",
            1,
        ),
        (
            "if (@(Compare-Object @($cargoCheckoutMetadataExclusions) $observedCargoCheckoutMetadata -SyncWindow 0).Count -ne 0) {\n"
            "        throw 'Cargo Git checkout metadata inventory differs from the reviewed nondeterministic bookkeeping set'\n"
            "    }",
            1,
        ),
        ("Cargo Git checkout metadata inventory differs from the reviewed nondeterministic bookkeeping set", 1),
        (
            "$cargoBookkeepingExclusions = @(\n"
            "        '.global-cache', '.package-cache', '.package-cache-mutate'\n"
            "    ) + @($cargoCheckoutMetadataExclusions)",
            1,
        ),
        ("'-C', 'link-arg=/Brepro'", 1),
        ("'-C', 'link-arg=/DEBUG:NONE'", 1),
        ("--remap-path-prefix=$remappedSource=/collide-o-scope", 1),
        ('"/d1trimfile:$source"', 1),
        ("rustup run 1.98.0 cargo auditable build --locked --offline --release --bin collide-o-scope", 1),
        ("canonical release directory contains a PDB despite /DEBUG:NONE", 1),
        ("Assert-PortableExecutableHasNoCodeView -Executable $executable", 1),
        ("release executable contains a builder-specific path", 1),
        ("avcodec-63.dll,avdevice-63.dll,avfilter-12.dll,avformat-63.dll,avutil-61.dll,ffmpeg=9.0.1,swresample-7.dll,swscale-10.dll", 1),
        ("release executable embeds an incomplete or unexpected BuildIdentity", 1),
        ("[Security.Cryptography.SHA256]::HashData($executableBytes)", 1),
        ("reviewed LLVM inputs changed during the build", 1),
        ("source identity changed during the canonical build", 1),
        ("[Security.Cryptography.CryptographicOperations]::FixedTimeEquals($ownerBytes, $ownerCheckBytes)", 1),
        ("contract = 'collide-windows-canonical-repro-v1'", 1),
        ("cargo_manifest_exclusions = @($cargoBookkeepingExclusions)", 1),
        ("cleanup_succeeded = $false", 1),
        ("$restoreFailure = $null", 1),
        ("Remove-Item -LiteralPath $canonicalRoot -Recurse -Force", 1),
        ("canonical build did not complete its cleanup and restoration transaction", 1),
        ("[CollideReproducibleNativePaths]::CreateNewDirectory($publishRoot)", 1),
        ("publication staging bytes differ from the verified canonical executable", 1),
        ("[IO.Directory]::Move($publishRoot, $outputTarget)", 1),
        ("published executable differs from the verified canonical bytes", 1),
        ("$result.cleanup_succeeded = $true", 1),
        ("Assert-NoReparsePoints -Path $canonicalRoot -Recurse", 2),
    )


def validate_reviewed_reproducible_build_digest(build_script: str) -> None:
    normalized = build_script.replace("\r\n", "\n").replace("\r", "\n")
    observed = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if observed != REVIEWED_REPRODUCIBLE_BUILD_SHA256:
        fail("reproducible build wrapper differs from its reviewed semantic contract")


def _unique_index(text: str, marker: str) -> int:
    if text.count(marker) != 1:
        fail(f"canonical reproducibility marker is not unique: {marker}")
    return text.index(marker)


def validate_canonical_reproducible_build(build_script: str) -> None:
    for fragment, expected_count in canonical_reproducible_build_fragments():
        if build_script.count(fragment) != expected_count:
            fail(f"canonical reproducible build contract changed: {fragment}")

    forbidden = (
        r"(?i)RUNNER_TEMP|GITHUB_WORKSPACE",
        r"(?i)\bsubst(?:\.exe)?\b|\bmklink\b|New-Item[^\n]*Junction",
        r"(?i)--network",
        r"(?i)\bif\s*\(\s*\$false\s*\)",
        r"(?i)\b(?:Set|New|Remove|Clear)-Variable\b",
        r"Copy-VerifiedTree\s+-Source\s+\$inputCargoSeed(?:\s|$)",
        r"Copy-VerifiedTree[^\n]*(?:registry\\src|git\\checkouts)",
        r"CARGO_TARGET_DIR'\s+-Value\s+\$outputTarget",
        r"\$target\s*=\s*\$outputTarget",
        r"(?i)\bMove-Item\b",
    )
    if any(re.search(pattern, build_script) for pattern in forbidden):
        fail("reproducible build contains a caller-derived stage or unreviewed route")

    helper_write = "[Environment]::SetEnvironmentVariable($Name, $Value, 'Process')"
    if build_script.count(helper_write) != 1:
        fail("process-environment helper is not singular")
    outside_helper = build_script.replace(helper_write, "", 1)
    approved_environment_read = "Get-ChildItem Env:"
    if outside_helper.count(approved_environment_read) != 2:
        fail("process-environment read inventory changed")
    outside_helper = outside_helper.replace(approved_environment_read, "", 2)
    if re.search(
        r"(?i)\$env:|\bEnv\s*:|SetEnvironmentVariable\s*\(",
        outside_helper,
    ):
        fail("reproducible build mutates process environment outside its checked helper")

    approved_writes = sorted(
        (
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_NOSYSTEM' -Value '1'",
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_NOSYSTEM' -Value '1'",
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_GLOBAL' -Value 'NUL'",
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_GLOBAL' -Value 'NUL'",
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_COUNT' -Value '0'",
            "Set-ProcessEnvironmentValue -Name 'GIT_CONFIG_COUNT' -Value '0'",
            "Set-ProcessEnvironmentValue -Name 'GIT_ATTR_NOSYSTEM' -Value '1'",
            "Set-ProcessEnvironmentValue -Name 'GIT_ATTR_NOSYSTEM' -Value '1'",
            "Set-ProcessEnvironmentValue -Name $environmentName -Value $initialGitEnvironment[$environmentName]",
            "Set-ProcessEnvironmentValue -Name 'CARGO_HOME' -Value $cargoHome",
            "Set-ProcessEnvironmentValue -Name 'CARGO_TARGET_DIR' -Value $target",
            "Set-ProcessEnvironmentValue -Name 'CARGO_NET_OFFLINE' -Value 'true'",
            "Set-ProcessEnvironmentValue -Name 'CARGO_INCREMENTAL' -Value '0'",
            "Set-ProcessEnvironmentValue -Name 'RUSTUP_TOOLCHAIN' -Value '1.98.0'",
            "Set-ProcessEnvironmentValue -Name 'COLLIDE_BUILD_GIT_SHA' -Value $GitSha.ToLowerInvariant()",
            "Set-ProcessEnvironmentValue -Name 'COLLIDE_BUILD_GIT_DIRTY' -Value 'false'",
            "Set-ProcessEnvironmentValue -Name 'COLLIDE_PUBLISHED_ARTIFACT' -Value 'true'",
            "Set-ProcessEnvironmentValue -Name 'FFMPEG_DIR' -Value $ffmpeg",
            "Set-ProcessEnvironmentValue -Name 'FFMPEG_VERSION' -Value '9.0.1'",
            "Set-ProcessEnvironmentValue -Name 'SOURCE_DATE_EPOCH' -Value $SourceDateEpoch",
            "Set-ProcessEnvironmentValue -Name 'TEMP' -Value $canonicalTemp",
            "Set-ProcessEnvironmentValue -Name 'TMP' -Value $canonicalTemp",
            "Set-ProcessEnvironmentValue -Name 'TMPDIR' -Value $canonicalTemp",
            "Set-ProcessEnvironmentValue -Name $controlledArchiverName -Value $llvmAr",
            "Set-ProcessEnvironmentValue -Name $controlledAwsCmakeName -Value '0'",
            "Set-ProcessEnvironmentValue -Name $controlledAwsSystemName -Value '0'",
            "Set-ProcessEnvironmentValue -Name 'CL' -Value $null",
            "Set-ProcessEnvironmentValue -Name '_CL_' -Value $null",
            "Set-ProcessEnvironmentValue -Name 'LINK' -Value $null",
            "Set-ProcessEnvironmentValue -Name '_LINK_' -Value $null",
            "Set-ProcessEnvironmentValue -Name 'PATH' -Value $buildPath",
            "Set-ProcessEnvironmentValue -Name 'PATH' -Value $buildPath",
            "Set-ProcessEnvironmentValue -Name 'CARGO_ENCODED_RUSTFLAGS' -Value $encodedFlags",
            "Set-ProcessEnvironmentValue -Name 'CC_SHELL_ESCAPED_FLAGS' -Value '1'",
            "Set-ProcessEnvironmentValue -Name 'CFLAGS' -Value $nativeTrimFlags",
            "Set-ProcessEnvironmentValue -Name 'CXXFLAGS' -Value $nativeTrimFlags",
            "Set-ProcessEnvironmentValue -Name 'PATH' -Value ((Join-Path $ffmpeg 'bin') + ';' + $buildPath)",
            "Set-ProcessEnvironmentValue -Name $environmentName -Value $saved[$environmentName]",
        )
    )
    observed_writes = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "Set-ProcessEnvironmentValue -Name " in line
    )
    if observed_writes != approved_writes:
        fail("reproducible build process-environment write inventory changed")

    approved_metadata_exclusion_lines = sorted(
        (
            "$cargoCheckoutMetadataExclusions = @()",
            "$cargoCheckoutMetadataExclusions += $relativeMetadata",
            "if (@(Compare-Object @($cargoCheckoutMetadataExclusions) $observedCargoCheckoutMetadata -SyncWindow 0).Count -ne 0) {",
            ") + @($cargoCheckoutMetadataExclusions)",
        )
    )
    observed_metadata_exclusion_lines = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "$cargoCheckoutMetadataExclusions" in line
    )
    if observed_metadata_exclusion_lines != approved_metadata_exclusion_lines:
        fail("Cargo checkout metadata exclusion inventory changed")

    approved_bookkeeping_exclusion_lines = sorted(
        (
            "$cargoBookkeepingExclusions = @(",
            "$cargoExpandedManifestSha256 = Get-TreeManifestDigest -Path $cargoHome -ExcludedRelativePaths $cargoBookkeepingExclusions",
            "cargo_manifest_exclusions = @($cargoBookkeepingExclusions)",
        )
    )
    observed_bookkeeping_exclusion_lines = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "$cargoBookkeepingExclusions" in line
    )
    if observed_bookkeeping_exclusion_lines != approved_bookkeeping_exclusion_lines:
        fail("Cargo bookkeeping exclusion use inventory changed")

    approved_checkout_review_lines = sorted(
        (
            "$cargoGitCheckoutReview = [ordered]@{",
            "foreach ($relativeCheckout in $cargoGitCheckoutReview.Keys) {",
            "if ($LASTEXITCODE -ne 0 -or $checkoutCommit -cne $cargoGitCheckoutReview[$relativeCheckout]) {",
        )
    )
    observed_checkout_review_lines = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "$cargoGitCheckoutReview" in line
    )
    if observed_checkout_review_lines != approved_checkout_review_lines:
        fail("Cargo Git checkout review-set use inventory changed")

    approved_relative_metadata_lines = sorted(
        (
            "$relativeMetadataFiles = @(",
            "foreach ($relativeMetadata in $relativeMetadataFiles) {",
        )
    )
    observed_relative_metadata_lines = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "$relativeMetadataFiles" in line
    )
    if observed_relative_metadata_lines != approved_relative_metadata_lines:
        fail("Cargo Git relative metadata-set use inventory changed")

    approved_observed_metadata_lines = sorted(
        (
            "$observedCargoCheckoutMetadata = @(",
            "if (@(Compare-Object @($cargoCheckoutMetadataExclusions) $observedCargoCheckoutMetadata -SyncWindow 0).Count -ne 0) {",
        )
    )
    observed_observed_metadata_lines = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "$observedCargoCheckoutMetadata" in line
    )
    if observed_observed_metadata_lines != approved_observed_metadata_lines:
        fail("observed Cargo Git metadata inventory use changed")

    recursive_deletes = sorted(
        line.strip()
        for line in build_script.splitlines()
        if "remove-item" in line.lower() and "-recurse" in line.lower()
    )
    if recursive_deletes != sorted(
        (
            "Remove-Item -LiteralPath $canonicalRoot -Recurse -Force",
            "Remove-Item -LiteralPath $publishRoot -Recurse -Force",
        )
    ):
        fail("reproducible build has an unreviewed recursive deletion")

    order = (
        "$resolvedSource = (Resolve-Path -LiteralPath $SourceRoot).Path",
        "$gitOverrideEnvironmentNames = @(",
        "$saved = @{}",
        "$mutex = [Threading.Mutex]::new($false, $canonicalMutexName)",
        "$mutexHeld = $mutex.WaitOne(0)",
        "canonical reproducible staging root already exists and will not be auto-deleted",
        "[CollideReproducibleNativePaths]::CreateNewDirectory($canonicalRoot)",
        "$ownerStream = [IO.File]::Open($ownerPath",
        "archive --format=tar --output=$sourceArchive",
        "Copy-VerifiedTree -Source (Join-Path $inputCargoSeed 'registry\\cache')",
        "foreach ($component in @('bin', 'include', 'lib'))",
        "Set-ProcessEnvironmentValue -Name 'CARGO_HOME' -Value $cargoHome",
        "rustup run 1.98.0 cargo fetch --locked --offline",
        "$cargoGitCheckoutReview = [ordered]@{",
        "$checkoutCommit = (git @gitSafetyArguments",
        "git @gitSafetyArguments -C $checkoutPath diff-index --quiet HEAD --",
        "$checkoutStatus = @(git @gitSafetyArguments",
        "$relativeMetadataFiles = @(",
        "$observedCargoCheckoutMetadata = @(",
        "Cargo Git checkout metadata inventory differs from the reviewed nondeterministic bookkeeping set",
        "$cargoBookkeepingExclusions = @(",
        "$cargoExpandedManifestSha256 = Get-TreeManifestDigest",
        "$encodedFlags = @(",
        "rustup run 1.98.0 cargo auditable build --locked --offline --release --bin collide-o-scope",
        "canonical release directory contains a PDB despite /DEBUG:NONE",
        "Assert-PortableExecutableHasNoCodeView -Executable $executable",
        "$builderSpecificPaths = @(",
        "$identity = $identityJson | ConvertFrom-Json",
        "$result = [ordered]@{",
        "$restoreFailure = $null",
        "Remove-Item -LiteralPath $canonicalRoot -Recurse -Force",
        "canonical build did not complete its cleanup and restoration transaction",
        "[CollideReproducibleNativePaths]::CreateNewDirectory($publishRoot)",
        "publication staging bytes differ from the verified canonical executable",
        "[IO.Directory]::Move($publishRoot, $outputTarget)",
        "$finalHash = (Get-FileHash -LiteralPath $finalExecutable",
        "$result.cleanup_succeeded = $true",
    )
    positions = [_unique_index(build_script, marker) for marker in order]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        fail("canonical build controls, cleanup, and publication are out of order")


def _expect_semantic_rejection(build_script: str) -> None:
    try:
        validate_canonical_reproducible_build(build_script)
    except ValueError:
        return
    fail("canonical reproducibility hostile mutation was not rejected")


def self_test_canonical_reproducible_build(build_script: str) -> None:
    validate_reviewed_reproducible_build_digest(build_script)
    validate_canonical_reproducible_build(build_script)
    for fragment, _ in canonical_reproducible_build_fragments():
        _expect_semantic_rejection(build_script.replace(fragment, "", 1))

    hostile_replacements = (
        ("$canonicalRoot = 'C:\\cosrepro'", "$canonicalRoot = Join-Path $env:RUNNER_TEMP 'cosrepro'"),
        (
            "throw 'canonical reproducible staging root already exists and will not be auto-deleted'",
            "Remove-Item -LiteralPath $canonicalRoot -Recurse -Force",
        ),
        ("[CollideReproducibleNativePaths]::CreateNewDirectory($canonicalRoot)", "New-Item -ItemType Directory -Path $canonicalRoot"),
        ("archive --format=tar --output=$sourceArchive", "Copy-VerifiedTree -Source $inputSource -Destination $canonicalSource"),
        ("$target = $canonicalTarget", "$target = $outputTarget"),
        ("'-C', 'link-arg=/DEBUG:NONE'", "'-C', 'link-arg=/DEBUG:FULL'"),
        ("rustup run 1.98.0 cargo fetch --locked --offline", "rustup run 1.98.0 cargo fetch --locked"),
        (
            "rustup run 1.98.0 cargo auditable build --locked --offline --release --bin collide-o-scope",
            "rustup run 1.98.0 cargo auditable build --locked --release --bin collide-o-scope",
        ),
        ("[IO.Directory]::Move($publishRoot, $outputTarget)", "Move-Item -LiteralPath $publishRoot -Destination $outputTarget"),
    )
    for original, replacement in hostile_replacements:
        _expect_semantic_rejection(build_script.replace(original, replacement, 1))

    build_marker = "rustup run 1.98.0 cargo auditable build --locked --offline --release --bin collide-o-scope"
    for injected in (
        "Set-Item Env:CL ambient",
        "Set-Content -Path Env:LINK -Value ambient",
        "Remove-Item Env:RUSTFLAGS",
        "Clear-Item Env:CARGO_HOME",
        "Copy-Item Env:RUST_LOG Env:RUSTFLAGS",
        "[Environment]::setenvironmentvariable('CL', 'ambient', 'Process')",
        "remove-item -LiteralPath C:\\unreviewed -recurse -Force",
    ):
        _expect_semantic_rejection(
            build_script.replace(build_marker, f"{injected}\n    {build_marker}", 1)
        )

    exclusion_marker = "$cargoBookkeepingExclusions = @("
    _expect_semantic_rejection(
        build_script.replace(
            exclusion_marker,
            "$cargoCheckoutMetadataExclusions += 'registry/src/unreviewed/Cargo.toml'\n"
            f"    {exclusion_marker}",
            1,
        )
    )
    manifest_marker = (
        "$cargoExpandedManifestSha256 = Get-TreeManifestDigest -Path $cargoHome "
        "-ExcludedRelativePaths $cargoBookkeepingExclusions"
    )
    _expect_semantic_rejection(
        build_script.replace(
            manifest_marker,
            "$cargoBookkeepingExclusions += 'registry/src/unreviewed/Cargo.toml'\n"
            f"    {manifest_marker}",
            1,
        )
    )
    diff_marker = "git @gitSafetyArguments -C $checkoutPath diff-index --quiet HEAD --"
    _expect_semantic_rejection(
        build_script.replace(
            diff_marker,
            f"if ($false) {{\n            {diff_marker}\n        }}",
            1,
        )
    )
    relative_metadata_block = (
        "$relativeMetadataFiles = @(\n"
        "            \"$relativeCheckout/.git/index\",\n"
        "            \"$relativeCheckout/.git/logs/HEAD\",\n"
        "            \"$relativeCheckout/.git/logs/refs/heads/master\"\n"
        "        )"
    )
    _expect_semantic_rejection(
        build_script.replace(
            relative_metadata_block,
            relative_metadata_block
            + '\n        $relativeMetadataFiles += "$relativeCheckout/vendor/extra/.git/index"',
            1,
        )
    )
    checkout_review_block = (
        "$cargoGitCheckoutReview = [ordered]@{\n"
        "        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500' = '4b79500dfac64efcfb393eebc89f5c75565ee5ae'\n"
        "        'git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx' = '5aa788d5134f577c23eba158ded7592c4c471050'\n"
        "    }"
    )
    _expect_semantic_rejection(
        build_script.replace(
            checkout_review_block,
            checkout_review_block
            + "\n    $cargoGitCheckoutReview['git/checkouts/third/deadbee'] = '0000000000000000000000000000000000000000'",
            1,
        )
    )
    observed_metadata_marker = "$observedCargoCheckoutMetadata = @("
    _expect_semantic_rejection(
        build_script.replace(
            observed_metadata_marker,
            "Set-Variable -Name observedCargoCheckoutMetadata -Value @()\n    "
            + observed_metadata_marker,
            1,
        )
    )

    env_marker = "Set-ProcessEnvironmentValue -Name 'CARGO_HOME' -Value $cargoHome"
    mutation = build_script.replace(build_marker, "__BUILD__", 1).replace(env_marker, build_marker, 1).replace("__BUILD__", env_marker, 1)
    _expect_semantic_rejection(mutation)

    cleanup_marker = "Remove-Item -LiteralPath $canonicalRoot -Recurse -Force"
    publish_marker = "[IO.Directory]::Move($publishRoot, $outputTarget)"
    mutation = build_script.replace(cleanup_marker, "__CLEANUP__", 1).replace(publish_marker, cleanup_marker, 1).replace("__CLEANUP__", publish_marker, 1)
    _expect_semantic_rejection(mutation)


def pinned_llvm_workflow_fragments() -> tuple[tuple[str, int], ...]:
    return (
        ("LLVM_VERSION: 22.1.8", 1),
        (
            "LLVM_WINDOWS_INSTALLER_SHA256: 16e5709785fef73c854646241c4a92c5cd574318d1b33c63330dd7721903e55c",
            1,
        ),
        ("LLVM_SOURCE_COMMIT: ca7933e47d3a3451d81e72ac174dcb5aa28b59d1", 1),
        (
            "LLVM_AR_SHA256: 80934e8f208a0cc2a87a6057f871d0f492461952b8672464749a6c3dff34109c",
            1,
        ),
        (
            "LIBCLANG_SHA256: 51fed10c43c3d31c1fe5bfe76bac60150970961e9b9b23cf014dbfcb5398bbfc",
            1,
        ),
        (
            "- name: Download and verify pinned LLVM toolchain\n"
            "        shell: pwsh\n"
            "        env:\n"
            "          GH_TOKEN: ${{ github.token }}",
            1,
        ),
        ('$destination = "C:\\collide-llvm-$env:LLVM_VERSION"', 1),
        ("Pinned LLVM destination already exists", 1),
        (
            "if (Test-Path -LiteralPath $destination) {\n"
            "            throw \"Pinned LLVM destination already exists\"\n"
            "          }",
            1,
        ),
        (
            'https://github.com/llvm/llvm-project/releases/download/llvmorg-$env:LLVM_VERSION/LLVM-$env:LLVM_VERSION-win64.exe',
            1,
        ),
        ("$installerHash -cne $env:LLVM_WINDOWS_INSTALLER_SHA256", 1),
        (
            "if ($installerHash -cne $env:LLVM_WINDOWS_INSTALLER_SHA256) {\n"
            "            throw \"LLVM installer checksum mismatch\"\n"
            "          }",
            1,
        ),
        ("gh attestation verify $installer `", 1),
        ("--repo llvm/llvm-project `", 1),
        (
            "--signer-workflow llvm/llvm-project/.github/workflows/release-binaries.yml `",
            1,
        ),
        ('--source-ref "refs/tags/llvmorg-$env:LLVM_VERSION" `', 1),
        ("--source-digest $env:LLVM_SOURCE_COMMIT `", 1),
        ("--deny-self-hosted-runners", 1),
        (
            "--deny-self-hosted-runners\n"
            "          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
            "          $sevenZip =",
            1,
        ),
        ("Get-Command 7z.exe -CommandType Application -ErrorAction Stop", 1),
        (
            "& $sevenZip x $installer \"-o$destination\" 'bin\\llvm-ar.exe' 'bin\\libclang.dll' -y",
            1,
        ),
        (
            "'bin\\libclang.dll' -y\n"
            "          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
            "          $llvmBin =",
            1,
        ),
        ("Pinned LLVM extraction contains unexpected files", 1),
        (
            "if ($actualFiles.Count -ne 2 -or (Compare-Object $expectedFiles $actualFiles)) {\n"
            "            throw \"Pinned LLVM extraction contains unexpected files\"\n"
            "          }",
            1,
        ),
        ("$llvmArHash -cne $env:LLVM_AR_SHA256", 1),
        (
            'if ($llvmArHash -cne $env:LLVM_AR_SHA256) { throw "llvm-ar.exe checksum mismatch" }',
            1,
        ),
        ("LLVM version $([regex]::Escape($env:LLVM_VERSION))", 1),
        (
            'if ($LASTEXITCODE -ne 0 -or $llvmArVersion -notmatch "(?m)^  LLVM version '
            '$([regex]::Escape($env:LLVM_VERSION))$") {\n'
            '            throw "llvm-ar.exe version mismatch"\n'
            "          }",
            1,
        ),
        ("$libclangHash -cne $env:LIBCLANG_SHA256", 1),
        (
            "if ($libclangHash -cne $env:LIBCLANG_SHA256 -or $libclangVersion -cne $env:LLVM_VERSION) {\n"
            "            throw \"libclang.dll identity mismatch\"\n"
            "          }",
            1,
        ),
        ('"LIBCLANG_PATH=$llvmBin" >> $env:GITHUB_ENV', 1),
    )


def validate_pinned_llvm_workflow(release: str) -> None:
    step_marker = "      - name: Download and verify pinned LLVM toolchain"
    if release.count(step_marker) != 1:
        fail("release workflow pinned LLVM step is absent or duplicated")
    step_start = release.index(step_marker)
    following_step = re.search(r"(?m)^      - name: ", release[step_start + len(step_marker) :])
    if following_step is None:
        fail("release workflow pinned LLVM step has no bounded successor")
    step_end = step_start + len(step_marker) + following_step.start()
    llvm_step = release[step_start:step_end]
    observed_step_digest = hashlib.sha256(llvm_step.encode("utf-8")).hexdigest()
    if observed_step_digest != REVIEWED_PINNED_LLVM_STEP_SHA256:
        fail("release workflow pinned LLVM step differs from its reviewed bytes")

    for fragment, expected_count in pinned_llvm_workflow_fragments():
        if release.count(fragment) != expected_count:
            fail(f"release workflow pinned LLVM contract changed: {fragment}")
    forbidden = (
        r"C:\\Program Files\\LLVM",
        r"(?i)\b(?:winget|choco)\s+install\s+(?:llvm|llvm\.llvm)\b",
        r"(?i)\bStart-Process\b|&\s*\$installer(?:\s|$)",
        r"(?i)Invoke-Expression|\biex\b",
    )
    if any(re.search(pattern, release) for pattern in forbidden):
        fail("release workflow executes or routes an unreviewed LLVM distribution")

    approved_installer_lines = sorted(
        (
            '$installer = Join-Path $env:RUNNER_TEMP "LLVM-$env:LLVM_VERSION-win64.exe"',
            "Invoke-WebRequest -Uri $url -OutFile $installer -MaximumRetryCount 3 -RetryIntervalSec 5",
            "$installerHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()",
            "gh attestation verify $installer `",
            "& $sevenZip x $installer \"-o$destination\" 'bin\\llvm-ar.exe' 'bin\\libclang.dll' -y",
        )
    )
    observed_installer_lines = sorted(
        line.strip()
        for line in release.splitlines()
        if re.search(r"(?i)\$(?:\{installer\}|installer(?![A-Za-z0-9_]))", line)
    )
    if observed_installer_lines != approved_installer_lines:
        fail("release workflow LLVM installer use inventory changed")

    order = (
        "- name: Download and verify pinned LLVM toolchain",
        "Invoke-WebRequest -Uri $url -OutFile $installer",
        "$installerHash = (Get-FileHash",
        "$installerHash -cne $env:LLVM_WINDOWS_INSTALLER_SHA256",
        "gh attestation verify $installer `",
        "--deny-self-hosted-runners\n          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n          $sevenZip =",
        "& $sevenZip x $installer",
        "'bin\\libclang.dll' -y\n          if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n          $llvmBin =",
        "$extractedFiles = @(Get-ChildItem",
        "$llvmArHash = (Get-FileHash",
        "$llvmArHash -cne $env:LLVM_AR_SHA256",
        "$llvmArVersion = (& $llvmAr --version)",
        "$libclangHash = (Get-FileHash",
        "$libclangHash -cne $env:LIBCLANG_SHA256",
        '"LIBCLANG_PATH=$llvmBin" >> $env:GITHUB_ENV',
        "- name: Download pinned FFmpeg shared runtime",
        "- name: Build twice from clean independent unequal roots",
    )
    positions = [_unique_index(release, marker) for marker in order]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        fail("release workflow does not attest, hash, extract, and verify LLVM in order")


def self_test_pinned_llvm_workflow(release: str) -> None:
    validate_pinned_llvm_workflow(release)
    for fragment, _ in pinned_llvm_workflow_fragments():
        mutation = release.replace(fragment, "", 1)
        try:
            validate_pinned_llvm_workflow(mutation)
        except ValueError:
            continue
        fail("pinned LLVM workflow fragment mutation was not rejected")

    hostile = (
        ("--repo llvm/llvm-project `", "--owner llvm `"),
        (
            "--signer-workflow llvm/llvm-project/.github/workflows/release-binaries.yml `",
            "--signer-repo llvm/llvm-project `",
        ),
        ("--deny-self-hosted-runners", "--limit 1"),
        (
            "$installerHash -cne $env:LLVM_WINDOWS_INSTALLER_SHA256",
            "$installerHash -eq $env:LLVM_WINDOWS_INSTALLER_SHA256",
        ),
        (
            "& $sevenZip x $installer \"-o$destination\" 'bin\\llvm-ar.exe' 'bin\\libclang.dll' -y",
            "& $installer /S",
        ),
        (
            '"LIBCLANG_PATH=$llvmBin" >> $env:GITHUB_ENV',
            '"LIBCLANG_PATH=C:\\Program Files\\LLVM\\bin" >> $env:GITHUB_ENV',
        ),
        (
            'throw "LLVM installer checksum mismatch"',
            'Write-Warning "LLVM installer checksum mismatch"',
        ),
        (
            'throw "Pinned LLVM destination already exists"',
            'Write-Warning "Pinned LLVM destination already exists"',
        ),
        (
            'throw "Pinned LLVM extraction contains unexpected files"',
            'Write-Warning "Pinned LLVM extraction contains unexpected files"',
        ),
        (
            'throw "llvm-ar.exe checksum mismatch"',
            'Write-Warning "llvm-ar.exe checksum mismatch"',
        ),
        (
            'throw "libclang.dll identity mismatch"',
            'Write-Warning "libclang.dll identity mismatch"',
        ),
    )
    for original, replacement in hostile:
        mutation = release.replace(original, replacement, 1)
        try:
            validate_pinned_llvm_workflow(mutation)
        except ValueError:
            continue
        fail(f"pinned LLVM hostile mutation was not rejected: {original}")

    execution_injection = release.replace(
        "$sevenZip = (Get-Command 7z.exe",
        "Start-Process $installer -Wait\n          $sevenZip = (Get-Command 7z.exe",
        1,
    )
    try:
        validate_pinned_llvm_workflow(execution_injection)
    except ValueError:
        pass
    else:
        fail("pinned LLVM checker accepted direct installer execution")

    braced_execution_injection = release.replace(
        "& $sevenZip x $installer",
        "Start-Process ${installer} -Wait\n          & $sevenZip x $installer",
        1,
    )
    try:
        validate_pinned_llvm_workflow(braced_execution_injection)
    except ValueError:
        pass
    else:
        fail("pinned LLVM checker accepted braced-variable installer execution")


def validate_release_checkout_isolation_and_pretag(release: str) -> None:
    exact = {
        "ref: ${{ github.event_name == 'workflow_dispatch' && github.sha || env.RELEASE_TAG }}": 1,
        'elif [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]; then': 1,
        'if [ -z "$DEFAULT_BRANCH" ] || [ "$GITHUB_REF" != "refs/heads/$DEFAULT_BRANCH" ]; then': 1,
        'remote_default_rows="$(git ls-remote --heads origin "refs/heads/$DEFAULT_BRANCH")"': 1,
        "Dispatched candidate is not the exact remote default-branch head": 1,
        'if [ "$RELEASE_TAG" != "v$version" ]; then': 1,
        'remote_rows="$(git ls-remote --tags origin "refs/tags/$RELEASE_TAG" "refs/tags/$RELEASE_TAG^{}")"': 1,
        "Pre-tag qualification requires the planned tag to be absent": 1,
        "python scripts/verify-release.py release-absent \\": 1,
        'echo "tag_object=" >> "$GITHUB_OUTPUT"': 1,
        "- name: Check out dependency-policy source": 1,
        "path: policy-source": 1,
        "persist-credentials: false": 5,
        "hashFiles('policy-source/Cargo.lock')": 1,
        "Push-Location policy-source": 1,
        "- name: Check out independent source A": 1,
        "- name: Check out independent source B": 1,
        "- name: Prove reproducibility checkouts are pristine": 1,
        "$env:GIT_CONFIG_NOSYSTEM = '1'": 2,
        "$env:GIT_CONFIG_GLOBAL = 'NUL'": 2,
        "$env:GIT_CONFIG_COUNT = '0'": 2,
        "$env:GIT_ATTR_NOSYSTEM = '1'": 2,
        "status --porcelain=v1 --untracked-files=all": 4,
        "if ($statusExit -ne 0) {": 1,
        "if ($dirty.Count -ne 0) {": 1,
        "Reproducibility checkout is not pristine: $source": 1,
        "Pre-tag qualification requires the planned tag to remain absent": 2,
        "- name: Revalidate pre-tag absence after qualification": 1,
        "if: github.event_name == 'workflow_dispatch'": 2,
        "Remote default-branch head changed during pre-tag qualification": 1,
        "steps: &windows_release_steps": 1,
        "steps: *windows_release_steps": 1,
    }
    if any(release.count(fragment) != count for fragment, count in exact.items()):
        fail("release pre-tag or checkout-isolation contract changed")

    resolve_marker = "      - name: Resolve one immutable candidate commit"
    resolve_start = release.index(resolve_marker)
    resolve_end = release.index("\n      - name:", resolve_start + 1)
    resolve_step = release[resolve_start:resolve_end]
    if (
        hashlib.sha256(resolve_step.encode("utf-8")).hexdigest()
        != REVIEWED_RELEASE_SOURCE_RESOLUTION_STEP_SHA256
    ):
        fail("release candidate resolution differs from its reviewed bytes")

    order = (
        "- name: Check out dependency-policy source",
        "- name: Enforce dependency, license, and vendor gates",
        "Push-Location policy-source",
        "- name: Check out independent source A",
        "- name: Check out independent source B",
        "- name: Validate tag and derive deterministic epoch",
        "- name: Prove reproducibility checkouts are pristine",
        "- name: Prepare unequal physical reproducibility roots",
        "- name: Build twice from clean independent unequal roots",
    )
    positions = [_unique_index(release, marker) for marker in order]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        fail("policy work and fresh reproducibility checkouts are out of order")

    gate_start = release.index("      - name: Enforce dependency, license, and vendor gates")
    gate_end = release.index("\n      - name:", gate_start + 1)
    gate = release[gate_start:gate_end]
    if "Push-Location policy-source" not in gate or "source-a" in gate or "source-b" in gate:
        fail("mutable policy tooling can contaminate a reproducibility checkout")

    pristine_start = release.index("      - name: Prove reproducibility checkouts are pristine")
    pristine_end = release.index("\n      - name:", pristine_start + 1)
    pristine = release[pristine_start:pristine_end]
    pristine_digest = hashlib.sha256(pristine.encode("utf-8")).hexdigest()
    if (
        pristine_digest != REVIEWED_PRISTINE_CHECKOUT_STEP_SHA256
        or pristine.count("git -c core.fsmonitor=false -c core.hooksPath=NUL -C $source") != 2
        or "Select-Object -First 32" not in pristine
        or ".Substring(0, 512) + '<truncated>'" not in pristine
        or "$boundedRows | ConvertTo-Json -Compress" not in pristine
        or "rows=$($dirty.Count) first_rows_json=$details" not in pristine
        or "--untracked-files=no" in pristine
        or "$dirty -join" in pristine
        or "git clean" in pristine
        or "git reset" in pristine
        or "Remove-Item" in pristine
    ):
        fail("fresh-checkout proof can hide, omit, or repair dirty state")

    cosign_marker = "      - name: Install pinned cosign"
    cosign_start = release.index(cosign_marker)
    cosign_end = release.index("\n      - name:", cosign_start + 1)
    pretag_cosign = release[cosign_start:cosign_end]
    if "\n        if:" in pretag_cosign:
        fail("pre-tag qualification skips the safe pinned-cosign installation proof")

    final_absence_marker = "      - name: Revalidate pre-tag absence after qualification"
    final_absence_start = release.index(final_absence_marker)
    final_absence_end = release.index("\n      - name:", final_absence_start + 1)
    final_absence = release[final_absence_start:final_absence_end]
    final_absence_digest = hashlib.sha256(final_absence.encode("utf-8")).hexdigest()
    final_absence_required = (
        "if: github.event_name == 'workflow_dispatch'",
        "DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}",
        '$env:GITHUB_REF -cne "refs/heads/$env:DEFAULT_BRANCH"',
        'git -C source-a ls-remote --heads origin "refs/heads/$env:DEFAULT_BRANCH"',
        'git -C source-a ls-remote --tags origin "refs/tags/$env:RELEASE_TAG"',
        "python source-a\\scripts\\verify-release.py release-absent `",
    )
    if (
        final_absence_digest != REVIEWED_PRETAG_FINAL_ABSENCE_STEP_SHA256
        or any(final_absence.count(fragment) != 1 for fragment in final_absence_required)
    ):
        fail("pre-tag completion can rely on stale branch, tag, or release state")

    qualification_tail = (
        "      - name: Assemble deterministic package and provenance",
        cosign_marker,
        final_absence_marker,
        "      - name: Keyless-sign the checksum manifest",
    )
    tail_positions = [release.index(marker) for marker in qualification_tail]
    if tail_positions != sorted(tail_positions):
        fail("final pre-tag absence check does not follow complete safe qualification")

    tag_only_condition = "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
    for step_name in (
        "Keyless-sign the checksum manifest",
        "Attest release evidence",
        "Upload release evidence",
        "Revalidate the live tag before publication",
        "Create release and upload initial assets once",
    ):
        marker = f"      - name: {step_name}"
        step_start = release.index(marker)
        step_end = release.find("\n      - name:", step_start + 1)
        if step_end < 0:
            step_end = len(release)
        if tag_only_condition not in release[step_start:step_end]:
            fail(f"workflow dispatch can execute tag-only step: {step_name}")

    redownload_header = (
        "  redownload-verify:\n"
        "    name: Verify draft and publish only after persistence checks\n"
        "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n"
    )
    if release.count(redownload_header) != 1:
        fail("workflow dispatch can enter the draft verification/publication job")

    tag_job_header = (
        "  reproduce-sign-publish:\n"
        "    name: Reproduce, attest, and sign\n"
        "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n"
    )
    qualify_job_header = (
        "  reproduce-qualify:\n"
        "    name: Qualify reproducible release without publication authority\n"
        "    if: github.event_name == 'workflow_dispatch'\n"
    )
    if release.count(tag_job_header) != 1 or release.count(qualify_job_header) != 1:
        fail("privileged tag job and read-only dispatch job are not event-exclusive")
    tag_job = release.split("  reproduce-sign-publish:\n", 1)[1].split(
        "  reproduce-qualify:\n", 1
    )[0]
    qualify_job = release.split("  reproduce-qualify:\n", 1)[1].split(
        "  redownload-verify:\n", 1
    )[0]
    tag_permissions = (
        "    permissions:\n"
        "      contents: write\n"
        "      id-token: write\n"
        "      attestations: write\n"
        "    steps: &windows_release_steps\n"
    )
    qualify_permissions = (
        "    permissions:\n"
        "      attestations: read\n"
        "      contents: read\n"
        "    steps: *windows_release_steps\n"
    )
    if tag_job.count(tag_permissions) != 1 or qualify_job.count(qualify_permissions) != 1:
        fail("release job permissions or anchored step ownership changed")
    if "write" in qualify_job or "id-token" in qualify_job or "outputs:" in qualify_job:
        fail("pre-tag qualification retains publication or identity-token authority")

    dispatch_start = release.index('elif [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]; then')
    dispatch_end = release.index("\n          else", dispatch_start + 1)
    dispatch = release[dispatch_start:dispatch_end]
    if (
        "release-absent" not in dispatch
        or "ls-remote --tags" not in dispatch
        or "tag_object=" not in dispatch
        or "gh release create" in dispatch
        or "gh release upload" in dispatch
        or "gh release edit" in dispatch
    ):
        fail("workflow dispatch can skip absence gates or mutate a release")


def self_test_release_checkout_isolation_and_pretag(release: str) -> None:
    validate_release_checkout_isolation_and_pretag(release)
    mutations = (
        release.replace("Push-Location policy-source", "Push-Location source-a", 1),
        release.replace(
            "ref: ${{ github.event_name == 'workflow_dispatch' && github.sha || env.RELEASE_TAG }}",
            "ref: ${{ env.RELEASE_TAG }}",
            1,
        ),
        release.replace("status --porcelain=v1 --untracked-files=all", "status --porcelain=v1 --untracked-files=no", 1),
        release.replace("if ($statusExit -ne 0) {", "if ($statusExit -lt 0) {", 1),
        release.replace("if ($dirty.Count -ne 0) {", "if ($dirty.Count -lt 0) {", 1),
        release.replace("$head -cne $expected", "$head -ceq $expected", 1),
        release.replace(
            "$dirty = @(git -c core.fsmonitor=false",
            "$dirty = @() # git -c core.fsmonitor=false",
            1,
        ),
        release.replace("python scripts/verify-release.py release-absent \\", "python scripts/verify-release.py verify \\", 1),
        release.replace('echo "tag_object=" >> "$GITHUB_OUTPUT"', 'echo "tag_object=unchecked" >> "$GITHUB_OUTPUT"', 1),
        release.replace("Pre-tag qualification requires the planned tag to be absent", "tag may already exist", 1),
        release.replace("persist-credentials: false", "persist-credentials: true", 1),
        release.replace(
            'if [ -z "$DEFAULT_BRANCH" ] || [ "$GITHUB_REF" != "refs/heads/$DEFAULT_BRANCH" ]; then',
            'if [ -z "$DEFAULT_BRANCH" ] || [ "$GITHUB_REF" = "refs/heads/$DEFAULT_BRANCH" ]; then',
            1,
        ),
        release.replace(
            "      - name: Revalidate pre-tag absence after qualification\n"
            "        if: github.event_name == 'workflow_dispatch'",
            "      - name: Revalidate pre-tag absence after qualification",
            1,
        ),
        release.replace(
            "python source-a\\scripts\\verify-release.py release-absent `",
            "python source-a\\scripts\\verify-release.py verify `",
            1,
        ),
        release.replace(
            "  redownload-verify:\n"
            "    name: Verify draft and publish only after persistence checks\n"
            "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n",
            "  redownload-verify:\n"
            "    name: Verify draft and publish only after persistence checks\n",
            1,
        ),
        release.replace(
            "  reproduce-sign-publish:\n"
            "    name: Reproduce, attest, and sign\n"
            "    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')\n",
            "  reproduce-sign-publish:\n"
            "    name: Reproduce, attest, and sign\n",
            1,
        ),
        release.replace(
            "      contents: read\n    steps: *windows_release_steps",
            "      contents: write\n    steps: *windows_release_steps",
            1,
        ),
        release.replace(
            "      contents: read\n    steps: *windows_release_steps",
            "      contents: read\n      id-token: write\n    steps: *windows_release_steps",
            1,
        ),
        release.replace("steps: *windows_release_steps", "steps: &windows_release_steps", 1),
        release.replace(
            "      - name: Keyless-sign the checksum manifest\n"
            "        if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
            "      - name: Keyless-sign the checksum manifest",
            1,
        ),
    )
    for mutation in mutations:
        try:
            validate_release_checkout_isolation_and_pretag(mutation)
        except ValueError:
            pass
        else:
            fail("release checkout-isolation or pre-tag mutation escaped policy")


def unequal_reproducibility_workflow_fragments() -> tuple[tuple[str, int], ...]:
    return (
        ("path: source-b-with-deliberately-different-path-length", 1),
        ("source-b-with-deliberately-different-path-length", 4),
        ("- name: Prepare unequal physical reproducibility roots", 1),
        ("$cargoA = Join-Path $env:RUNNER_TEMP 'ca'", 1),
        ("cargo-seed-b-with-deliberately-different-physical-path-length", 1),
        ("$targetA = Join-Path $env:RUNNER_TEMP 'ta'", 1),
        ("target-b-with-deliberately-different-physical-path-length", 1),
        ("$robocopyExit = $LASTEXITCODE", 1),
        (
            'if ($robocopyExit -ge 8) { throw "Cargo seed copy failed with exit code $robocopyExit" }',
            1,
        ),
        ("$global:LASTEXITCODE = 0", 1),
        ("Copy-SeedTree (Join-Path $defaultCargoHome 'registry\\cache')", 1),
        ("Copy-SeedTree (Join-Path $defaultCargoHome 'registry\\index')", 1),
        ("Copy-SeedTree (Join-Path $defaultCargoHome 'git\\db')", 1),
        ("Independent Cargo seed manifests differ", 1),
        ("A/B path lengths must deliberately differ", 1),
        ("REPRO_CARGO_SEED_SHA256=$manifestA", 1),
        ("- name: Build twice from clean independent unequal roots", 1),
        ("[Environment]::SetEnvironmentVariable('CARGO_HOME', $CargoSeed, 'Process')", 1),
        ("[Environment]::SetEnvironmentVariable('CARGO_HOME', $priorCargoHome, 'Process')", 1),
        ("Reproducible wrapper did not emit exactly one JSON receipt", 1),
        (
            "(($receipt.cargo_manifest_exclusions -join ',') -cne '.global-cache,.package-cache,.package-cache-mutate,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/.git/index,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/.git/logs/HEAD,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/.git/logs/refs/heads/master,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx/.git/index,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx/.git/logs/HEAD,"
            "git/checkouts/ntsc-rs-5808ee35e7b6c97f/4b79500/crates/openfx-plugin/vendor/openfx/.git/logs/refs/heads/master')",
            1,
        ),
        ("$receiptA = Invoke-ReproducibleBuild $env:REPRO_SOURCE_A $env:REPRO_TARGET_A $env:REPRO_CARGO_A", 1),
        ("$receiptB = Invoke-ReproducibleBuild $env:REPRO_SOURCE_B $env:REPRO_TARGET_B $env:REPRO_CARGO_B", 1),
        ("Independent build receipts differ at $property", 1),
        ('--executable "$env:REPRO_TARGET_A\\release\\collide-o-scope.exe"', 1),
        ('--second-executable "$env:REPRO_TARGET_B\\release\\collide-o-scope.exe"', 1),
    )


def validate_unequal_reproducibility_workflow(release: str) -> None:
    reviewed_steps = (
        (
            "Prepare unequal physical reproducibility roots",
            REVIEWED_UNEQUAL_PREPARE_STEP_SHA256,
        ),
        (
            "Build twice from clean independent unequal roots",
            REVIEWED_UNEQUAL_BUILD_STEP_SHA256,
        ),
    )
    for step_name, expected_digest in reviewed_steps:
        marker = f"      - name: {step_name}"
        if release.count(marker) != 1:
            fail(f"reviewed unequal-root workflow step is not unique: {step_name}")
        step_start = release.index(marker)
        step_end = release.find("\n      - name:", step_start + 1)
        if step_end < 0:
            step_end = len(release)
        observed_digest = hashlib.sha256(
            release[step_start:step_end].encode("utf-8")
        ).hexdigest()
        if observed_digest != expected_digest:
            fail(f"reviewed unequal-root workflow step changed: {step_name}")

    prepare_marker = "      - name: Prepare unequal physical reproducibility roots"
    build_marker = "      - name: Build twice from clean independent unequal roots"
    successor_marker = "      - name: Generate reproducible CycloneDX SBOM"
    if release.count(successor_marker) != 1:
        fail("reviewed unequal-root workflow successor is not unique")
    region_start = release.index(prepare_marker)
    build_start = release.index(build_marker, region_start + 1)
    region_end = release.index(successor_marker, build_start + 1)
    observed_region_digest = hashlib.sha256(
        release[region_start:region_end].encode("utf-8")
    ).hexdigest()
    if observed_region_digest != REVIEWED_UNEQUAL_CONTIGUOUS_REGION_SHA256:
        fail("reviewed unequal-root workflow region is not contiguous")

    for fragment, expected_count in unequal_reproducibility_workflow_fragments():
        if release.count(fragment) != expected_count:
            fail(f"release workflow unequal-root contract changed: {fragment}")
    if (
        "GITHUB_PATH" in release
        or "RUNNER_TEMP\\target-a" in release
        or "RUNNER_TEMP\\target-b" in release
        or re.search(r"(?m)^\s+path:\s+source-b\s*$", release)
        or re.search(r"\$cargoB\s*=\s*\$cargoA|\$targetB\s*=\s*\$targetA", release)
        or 'second-executable "$env:REPRO_TARGET_A' in release
    ):
        fail("release workflow can collapse or contaminate independent build roots")

    order = (
        "path: source-a",
        "path: source-b-with-deliberately-different-path-length",
        "- name: Prepare unequal physical reproducibility roots",
        "& robocopy.exe $Source $Destination",
        "$robocopyExit = $LASTEXITCODE",
        'if ($robocopyExit -ge 8) { throw "Cargo seed copy failed with exit code $robocopyExit" }',
        "$global:LASTEXITCODE = 0",
        "$cargoA = Join-Path $env:RUNNER_TEMP 'ca'",
        "$cargoB = Join-Path $env:RUNNER_TEMP",
        "$manifestA = Get-ManifestDigest $cargoA",
        "- name: Build twice from clean independent unequal roots",
        "$receiptA = Invoke-ReproducibleBuild",
        "Build A left the canonical root present",
        "$receiptB = Invoke-ReproducibleBuild",
        "Independent build receipts differ at $property",
        '--executable "$env:REPRO_TARGET_A\\release\\collide-o-scope.exe"',
        '--second-executable "$env:REPRO_TARGET_B\\release\\collide-o-scope.exe"',
    )
    positions = [_unique_index(release, marker) for marker in order]
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        fail("release workflow does not create, build, compare, and package unequal roots in order")


def self_test_unequal_reproducibility_workflow(release: str) -> None:
    validate_unequal_reproducibility_workflow(release)
    for fragment, _ in unequal_reproducibility_workflow_fragments():
        mutation = release.replace(fragment, "", 1)
        try:
            validate_unequal_reproducibility_workflow(mutation)
        except ValueError:
            continue
        fail("unequal-root workflow fragment mutation was not rejected")

    hostile = (
        ("source-b-with-deliberately-different-path-length", "source-b"),
        ("$cargoB = Join-Path $env:RUNNER_TEMP", "$cargoB = $cargoA # "),
        ("$targetB = Join-Path $env:RUNNER_TEMP", "$targetB = $targetA # "),
        ("$robocopyExit = $LASTEXITCODE", "$robocopyExit = 0 # $LASTEXITCODE"),
        ("if ($robocopyExit -ge 8)", "if ($robocopyExit -ge 16)"),
        ("$global:LASTEXITCODE = 0", "$global:LASTEXITCODE = $robocopyExit"),
        ("Independent Cargo seed manifests differ", "Cargo seed mismatch ignored"),
        (
            "if ($manifestA -cne $manifestB) { throw 'Independent Cargo seed manifests differ' }",
            "if ($manifestA -cne $manifestB -and $false) { throw 'Independent Cargo seed manifests differ' }",
        ),
        (
            "if ($left -cne $right) { throw \"Independent build receipts differ at $property\" }",
            "if ($left -cne $right -and $false) { throw \"Independent build receipts differ at $property\" }",
        ),
        (
            '--second-executable "$env:REPRO_TARGET_B\\release\\collide-o-scope.exe"',
            '--second-executable "$env:REPRO_TARGET_A\\release\\collide-o-scope.exe"',
        ),
    )
    for original, replacement in hostile:
        mutation = release.replace(original, replacement, 1)
        try:
            validate_unequal_reproducibility_workflow(mutation)
        except ValueError:
            continue
        fail(f"unequal-root workflow hostile mutation was not rejected: {original}")

    receipt_exclusion_condition = next(
        fragment
        for fragment, _ in unequal_reproducibility_workflow_fragments()
        if fragment.startswith("(($receipt.cargo_manifest_exclusions -join ',')")
    )
    mutation = release.replace(
        receipt_exclusion_condition,
        f"({receipt_exclusion_condition} -and $false)",
        1,
    )
    try:
        validate_unequal_reproducibility_workflow(mutation)
    except ValueError:
        pass
    else:
        fail("unequal-root workflow accepted a neutralized receipt exclusion guard")

    prepare_to_build_injection = (
        "      - name: Collapse independent Cargo roots\n"
        "        shell: pwsh\n"
        "        run: |\n"
        '          "REPRO_CARGO_B=$env:REPRO_CARGO_A" >> $env:GITHUB_ENV\n\n'
    )
    for insertion_marker in (
        "      - name: Build twice from clean independent unequal roots",
        "      - name: Generate reproducible CycloneDX SBOM",
    ):
        mutation = release.replace(
            insertion_marker,
            prepare_to_build_injection + insertion_marker,
            1,
        )
        try:
            validate_unequal_reproducibility_workflow(mutation)
        except ValueError:
            continue
        fail("unequal-root workflow accepted an interstitial root-collapse step")


def validate_reproducible_sbom_workflow(release: str) -> None:
    marker = "      - name: Generate reproducible CycloneDX SBOM"
    successor = "      - name: Assemble deterministic package and provenance"
    if release.count(marker) != 1 or release.count(successor) != 1:
        fail("reviewed reproducible SBOM workflow boundary is not unique")
    step_start = release.index(marker)
    step_end = release.index(successor, step_start + 1)
    step = release[step_start:step_end]
    observed_digest = hashlib.sha256(step.encode("utf-8")).hexdigest()
    required = (
        '$ErrorActionPreference = "Stop"',
        '$env:SOURCE_DATE_EPOCH = "${{ steps.source.outputs.epoch }}"',
        "$env:GIT_CONFIG_NOSYSTEM = '1'",
        "$env:GIT_CONFIG_GLOBAL = 'NUL'",
        "$env:GIT_CONFIG_COUNT = '0'",
        "$env:GIT_ATTR_NOSYSTEM = '1'",
        "$expectedName = 'collide-o-scope.cdx.json'",
        "$normalizedA = Join-Path $env:RUNNER_TEMP $expectedName",
        "$normalizedB = Join-Path $env:RUNNER_TEMP 'collide-o-scope-b.cdx.json'",
        "$policyScript = Join-Path $env:REPRO_SOURCE_A 'scripts\\cyclonedx_sbom.py'",
        "foreach ($path in @($normalizedA, $normalizedB))",
        "python $policyScript self-test",
        "function New-NormalizedSbom",
        "[Parameter(Mandatory = $true)][string]$SourceLiteral",
        "$sourceItem = Get-Item -LiteralPath $SourceLiteral -Force",
        "$null -ne $sourceItem.LinkType",
        "($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0",
        "$before = @(git -c core.fsmonitor=false -c core.hooksPath=NUL -C $source status --porcelain=v1 --untracked-files=all)",
        "$beforeExit = $LASTEXITCODE",
        '$beforeExit -ne 0 -or $before.Count -ne 0',
        "$preexisting = @(Get-ChildItem -LiteralPath $source -Force | Where-Object { $_.Name -like 'collide-o-scope.cdx*' })",
        '$preexisting.Count -ne 0',
        "Push-Location $source",
        "try {",
        "cargo cyclonedx --format json --spec-version 1.5 --override-filename collide-o-scope.cdx",
        "$cargoExit = $LASTEXITCODE",
        "} finally {",
        "Pop-Location",
        "if ($cargoExit -ne 0) { exit $cargoExit }",
        "$outputs = @(Get-ChildItem -LiteralPath $source -Force | Where-Object { $_.Name -like 'collide-o-scope.cdx*' })",
        "$outputs.Count -ne 1",
        "$outputs[0].Name -cne $expectedName",
        "$outputs[0].PSIsContainer",
        "$null -ne $outputs[0].LinkType",
        "($outputs[0].Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0",
        "$after = @(git -c core.fsmonitor=false -c core.hooksPath=NUL -C $source status --porcelain=v1 --untracked-files=all)",
        "$afterExit = $LASTEXITCODE",
        '$after[0] -cne "?? $expectedName"',
        "python $policyScript normalize `",
        "--input $outputs[0].FullName",
        "--source-root $source",
        "--output $Normalized",
        '--source-date-epoch "${{ steps.source.outputs.epoch }}"',
        '--commit "${{ steps.source.outputs.commit }}"',
        "$normalizedItem = Get-Item -LiteralPath $Normalized -Force",
        "$normalizedItem.Name -cne [IO.Path]::GetFileName($Normalized)",
        "$normalizedItem.PSIsContainer",
        "$null -ne $normalizedItem.LinkType",
        "($normalizedItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0",
        "$sbomA = New-NormalizedSbom $env:REPRO_SOURCE_A $normalizedA 'A'",
        "$sbomB = New-NormalizedSbom $env:REPRO_SOURCE_B $normalizedB 'B'",
        "$bytesA = [IO.File]::ReadAllBytes($sbomA.Normalized)",
        "$bytesB = [IO.File]::ReadAllBytes($sbomB.Normalized)",
        "$sbomA.Length -ne $sbomB.Length",
        "$sbomA.Sha256 -cne $sbomB.Sha256",
        "[Convert]::ToBase64String($bytesA) -cne [Convert]::ToBase64String($bytesB)",
        "throw 'Independent unequal-root normalized SBOMs are not byte-identical'",
        "Remove-Item -LiteralPath $sbomA.Raw",
        "Remove-Item -LiteralPath $sbomB.Raw",
        "foreach ($source in @($env:REPRO_SOURCE_A, $env:REPRO_SOURCE_B))",
        "$final = @(git -c core.fsmonitor=false -c core.hooksPath=NUL -C $source status --porcelain=v1 --untracked-files=all)",
        "$finalExit = $LASTEXITCODE",
        "$finalExit -ne 0 -or $final.Count -ne 0",
    )
    downstream = release[step_end : release.find("\n      - name:", step_end + 1)]
    if (
        observed_digest != REVIEWED_REPRODUCIBLE_SBOM_STEP_SHA256
        or any(step.count(fragment) != 1 for fragment in required)
        or step.count("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }") != 2
        or release.count('  PYTHONDONTWRITEBYTECODE: "1"') != 1
        or "--override-filename collide-o-scope.cdx.json" in step
        or "ContinueOnError" in step
        or downstream.count('--sbom "$env:RUNNER_TEMP\\collide-o-scope.cdx.json" `') != 1
    ):
        fail("reproducible SBOM generation differs from its reviewed output contract")


def self_test_reproducible_sbom_workflow(release: str) -> None:
    validate_reproducible_sbom_workflow(release)
    marker = "      - name: Generate reproducible CycloneDX SBOM"
    successor = "      - name: Assemble deterministic package and provenance"
    step_start = release.index(marker)
    step_end = release.index(successor, step_start + 1)
    step = release[step_start:step_end]
    hostile = (
        (
            "--override-filename collide-o-scope.cdx",
            "--override-filename collide-o-scope.cdx.json",
        ),
        ("--spec-version 1.5", "--spec-version 1.4"),
        (
            "$normalizedB = Join-Path $env:RUNNER_TEMP 'collide-o-scope-b.cdx.json'",
            "$normalizedB = $normalizedA",
        ),
        (
            "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            "if ($LASTEXITCODE -ne 0 -and $false) { exit $LASTEXITCODE }",
        ),
        (
            "$preexisting.Count -ne 0",
            "if ($preexisting.Count -lt 0)",
        ),
        (
            "$outputs.Count -ne 1",
            "$outputs.Count -lt 1",
        ),
        (
            "$after.Count -ne 1",
            "$after.Count -lt 1",
        ),
        (
            "python $policyScript self-test",
            "Write-Output 'SBOM policy self-test skipped'",
        ),
        (
            "$sbomB = New-NormalizedSbom $env:REPRO_SOURCE_B $normalizedB 'B'",
            "$sbomB = New-NormalizedSbom $env:REPRO_SOURCE_A $normalizedB 'B'",
        ),
        (
            "[Convert]::ToBase64String($bytesA) -cne [Convert]::ToBase64String($bytesB)",
            "$false",
        ),
        (
            '--commit "${{ steps.source.outputs.commit }}"',
            '--commit "0000000000000000000000000000000000000000"',
        ),
        (
            "Remove-Item -LiteralPath $sbomB.Raw",
            "Write-Output 'raw B retained'",
        ),
    )
    for original, replacement in hostile:
        mutated_step = step.replace(original, replacement, 1)
        if mutated_step == step:
            fail(f"reproducible SBOM self-test fixture is absent: {original}")
        mutation = release[:step_start] + mutated_step + release[step_end:]
        try:
            validate_reproducible_sbom_workflow(mutation)
        except ValueError:
            continue
        fail(f"reproducible SBOM hostile mutation was not rejected: {original}")

    bytecode_mutation = release.replace(
        '  PYTHONDONTWRITEBYTECODE: "1"',
        '  PYTHONDONTWRITEBYTECODE: "0"',
        1,
    )
    try:
        validate_reproducible_sbom_workflow(bytecode_mutation)
    except ValueError:
        pass
    else:
        fail("reproducible SBOM policy accepted source-tree bytecode writes")

    downstream_mutation = release.replace(
        '--sbom "$env:RUNNER_TEMP\\collide-o-scope.cdx.json" `',
        "--sbom source-a\\collide-o-scope.cdx.json `",
        1,
    )
    try:
        validate_reproducible_sbom_workflow(downstream_mutation)
    except ValueError:
        pass
    else:
        fail("reproducible SBOM downstream-path mutation was not rejected")


def validate_reviewed_sbom_policy_digest(policy: str) -> None:
    if (
        hashlib.sha256(policy.encode("utf-8")).hexdigest()
        != REVIEWED_SBOM_POLICY_SHA256
        or 'EXPECTED_REPOSITORY_URL = "https://github.com/Spacejunk-io/collide-o-scope"' not in policy
        or 'return f"{EXPECTED_REPOSITORY_URL}/tree/{commit}"' not in policy
        or 'EXPECTED_DEPENDENCY_EDGES = 885' not in policy
        or 'EXPECTED_ROOT_EDGES = 36' not in policy
        or 'EXPECTED_REWRITTEN_REFERENCES = 13' not in policy
        or 'EXPECTED_SEMANTIC_PROFILE_SHA256 = (' not in policy
        or '"1605dc0f7f64c42735728495f8a42f85cfb8613c9606311fbe58608a4841ce00"' not in policy
        or 'object_pairs_hook=_reject_duplicate_keys' not in policy
        or 'parse_constant=_reject_nonfinite_constant' not in policy
        or 'if observed_changes != changed_paths:' not in policy
        or 'def semantic_profile_digest(' not in policy
        or 'if observed_semantic_sha256 != expected_semantic_sha256:' not in policy
        or 'fail("SBOM string uses excessive nested percent-encoding")' not in policy
        or '"substituted-component"' not in policy
        or 'replacement_edge' not in policy
        or "def validate_reference_graph(" not in policy
        or "def validate_normalized_sbom(" not in policy
        or "def self_test()" not in policy
    ):
        fail("CycloneDX SBOM policy differs from its reviewed bytes or contract")


def validate_adversarial_patch_trigger(adversarial: str) -> None:
    """Require every production patch parser/schema change to run fuzz CI."""
    try:
        trigger = adversarial.split("\npermissions:", 1)[0]
        push = trigger.split("  push:\n", 1)[1].split("  pull_request:\n", 1)[0]
        pull_request = trigger.split("  pull_request:\n", 1)[1].split(
            "  workflow_dispatch:\n", 1
        )[0]
    except IndexError as error:
        raise ValueError("adversarial workflow trigger structure is unexpected") from error

    patch_glob = '      - "src/patch/**"'
    if push.count(patch_glob) != 1 or pull_request.count(patch_glob) != 1:
        fail(
            "adversarial push and pull-request filters must each cover all src/patch paths"
        )


def self_test_adversarial_patch_trigger(adversarial: str) -> None:
    validate_adversarial_patch_trigger(adversarial)
    patch_glob = '      - "src/patch/**"'
    narrowed = '      - "src/patch/editor.rs"'
    pull_head, pull_tail = adversarial.rsplit(patch_glob, 1)
    for mutation in (
        adversarial.replace(patch_glob, narrowed, 1),
        pull_head + narrowed + pull_tail,
    ):
        try:
            validate_adversarial_patch_trigger(mutation)
        except ValueError:
            pass
        else:
            fail(
                "narrowed adversarial patch path escaped the workflow policy gate"
            )


def validate_shared_sbom_verifier(verifier: str) -> None:
    required = (
        "from cyclonedx_sbom import (",
        "SbomPolicyError,",
        "read_json as read_cyclonedx_json,",
        "self_test as self_test_cyclonedx_policy,",
        "validate_normalized_sbom,",
        "def validate_sbom(",
        "return validate_normalized_sbom(",
        "package_name=SBOM_PACKAGE_NAME,",
        "package_version=version,",
        "commit=commit.lower(),",
        "source_date_epoch=source_date_epoch,",
        'raise ReleaseError(f"SBOM release-profile validation failed: {error}")',
        "sbom = read_cyclonedx_json(args.sbom)",
        "sbom = read_cyclonedx_json(sbom_path)",
        "self_test_cyclonedx_policy()",
        'lambda: validate_sbom({}, "9.8.7", commit, 1_700_000_000)',
    )
    prepare_call = (
        "validate_sbom(\n"
        "        sbom,\n"
        '        identity["version"],\n'
        "        args.commit,\n"
        "        args.source_date_epoch,\n"
        "    )"
    )
    verify_call = (
        'validate_sbom(sbom, identity["version"], args.commit, commit_epoch)'
    )
    if (
        any(verifier.count(fragment) != 1 for fragment in required)
        or verifier.count(prepare_call) != 1
        or verifier.count(verify_call) != 1
        or verifier.count("except SbomPolicyError as error:") != 3
        or 'sbom.get("bomFormat")' in verifier
    ):
        fail("release prepare/verify do not share the reviewed SBOM policy")


def self_test_shared_sbom_verifier(verifier: str) -> None:
    validate_shared_sbom_verifier(verifier)
    mutations = (
        verifier.replace(
            "sbom = read_cyclonedx_json(args.sbom)",
            "sbom = read_json(args.sbom)",
            1,
        ),
        verifier.replace(
            "sbom = read_cyclonedx_json(sbom_path)",
            "sbom = read_json(sbom_path)",
            1,
        ),
        verifier.replace("commit=commit.lower(),", "commit='0' * 40,", 1),
        verifier.replace(
            "        args.source_date_epoch,\n    )",
            "        0,\n    )",
            1,
        ),
        verifier.replace(
            'validate_sbom(sbom, identity["version"], args.commit, commit_epoch)',
            'validate_sbom(sbom, identity["version"], args.commit, 0)',
            1,
        ),
        verifier.replace(
            "self_test_cyclonedx_policy()",
            "pass  # shared SBOM policy self-test skipped",
            1,
        ),
    )
    for mutation in mutations:
        if mutation == verifier:
            fail("shared SBOM verifier self-test fixture is absent")
        try:
            validate_shared_sbom_verifier(mutation)
        except ValueError:
            continue
        fail("shared SBOM verifier mutation escaped the policy gate")


def validate_pretag_package_tag_state(release: str, verifier: str) -> None:
    if hashlib.sha256(verifier.encode("utf-8")).hexdigest() != REVIEWED_RELEASE_VERIFIER_SHA256:
        fail("release verifier differs from its reviewed bytes")
    assembly_marker = "      - name: Assemble deterministic package and provenance"
    assembly_successor = "      - name: Install pinned cosign"
    if release.count(assembly_marker) != 1 or assembly_successor not in release:
        fail("reviewed release package-assembly workflow boundary is not unique")
    assembly_start = release.index(assembly_marker)
    assembly_end = release.index(assembly_successor, assembly_start + 1)
    assembly_step = release[assembly_start:assembly_end]
    if (
        hashlib.sha256(assembly_step.encode("utf-8")).hexdigest()
        != REVIEWED_PACKAGE_ASSEMBLY_STEP_SHA256
    ):
        fail("release package-assembly workflow differs from its reviewed bytes")
    tag_state_assignment = (
        "$tagState = if ($env:GITHUB_EVENT_NAME -ceq 'push') {\n"
        "            'annotated'\n"
        "          } elseif ($env:GITHUB_EVENT_NAME -ceq 'workflow_dispatch') {\n"
        "            'absent'\n"
        "          } else {\n"
        "            throw 'Unsupported release workflow event'\n"
        "          }"
    )
    if (
        release.count(tag_state_assignment) != 1
        or release.count("--tag-state $tagState") != 2
        or release.count("--tag-state annotated") != 3
        or "--tag-state absent" in release
    ):
        fail("release package verification does not bind explicit pre-tag/tagged state")
    verifier_contract = (
        "def local_ref_sha(reference: str) -> str | None:",
        '"rev-parse",',
        '"--verify",',
        '"--quiet",',
        "if completed.returncode == 1 and not stdout and not stderr:",
        "def validate_resolved_tag_state(",
        'if tag_state not in {"absent", "annotated"}:',
        'if tag_state == "absent":',
        'local_tag_type != "tag"',
        'raise ReleaseError("pre-tag qualification requires the local release tag to be absent")',
        'raise ReleaseError("tagged release verification requires the exact annotated tag")',
        "def validate_tag_binding(tag: str, commit: str, tag_state: str) -> None:",
        'git_text("cat-file", "-t", tag_ref)',
        'git_text("rev-parse", f"{tag_ref}^{{commit}}").lower()',
        "def validate_identity(identity: dict, tag: str, commit: str, tag_state: str) -> None:",
        "validate_tag_binding(tag, commit, tag_state)",
        "validate_identity(identity, args.tag, args.commit, args.tag_state)",
        'add_argument("--tag-state", choices=("absent", "annotated"), required=True)',
        'validate_resolved_tag_state("absent", None, None, None, commit)',
        'validate_resolved_tag_state("annotated", "2" * 40, "tag", commit, commit)',
        "def validate_signature_tag_state(require_signature: bool, tag_state: str) -> None:",
        'if require_signature and tag_state != "annotated":',
        "validate_signature_tag_state(args.require_signature, args.tag_state)",
        'lambda: validate_signature_tag_state(True, "absent")',
        "license_path: Path | None = None,",
        "readme_path: Path | None = None,",
        'license_path=extracted / "LICENSES/FFmpeg-GPL-3.0-or-later.txt",',
        'readme_path=extracted / "FFMPEG-README.txt",',
    )
    expected_counts = {
        "validate_identity(identity, args.tag, args.commit, args.tag_state)": 2,
        'add_argument("--tag-state", choices=("absent", "annotated"), required=True)': 2,
    }
    tag_binding_parts = verifier.split(
        "def validate_tag_binding(tag: str, commit: str, tag_state: str) -> None:",
        1,
    )
    tag_binding = (
        ""
        if len(tag_binding_parts) != 2
        else tag_binding_parts[1].split("\n\n\ndef validate_identity(", 1)[0]
    )
    verify_prefix = (
        "def verify(args: argparse.Namespace) -> dict:\n"
        "    validate_signature_tag_state(args.require_signature, args.tag_state)\n"
        "    directory = args.directory.resolve()"
    )
    ffmpeg_override_defaults = (
        '    if license_path is None:\n'
        '        license_path = root / "LICENSE"\n'
        '    if readme_path is None:\n'
        '        readme_path = root / "README.txt"\n'
        '    if not license_path.is_file() or not readme_path.is_file():'
    )
    if (
        any(fragment not in verifier for fragment in verifier_contract)
        or any(verifier.count(fragment) != count for fragment, count in expected_counts.items())
        or tag_binding.count('git_text("cat-file", "-t", tag_ref)') != 1
        or tag_binding.count('git_text("rev-parse", f"{tag_ref}^{{commit}}").lower()') != 1
        or tag_binding.count("local_tag_sha = local_ref_sha(tag_ref)") != 1
        or verifier.count(verify_prefix) != 1
        or verifier.count(ffmpeg_override_defaults) != 1
    ):
        fail("release verifier lacks the fail-closed absent/annotated tag-state contract")


def self_test_pretag_package_tag_state(release: str, verifier: str) -> None:
    validate_pretag_package_tag_state(release, verifier)
    release_mutations = (
        release.replace("--tag-state $tagState", "--tag-state annotated", 1),
        release.replace("--tag-state annotated", "--tag-state absent", 1),
        release.replace("-ceq 'workflow_dispatch'", "-ceq 'push'", 1),
        release.replace("throw 'Unsupported release workflow event'", "'absent'", 1),
    )
    for mutation in release_mutations:
        try:
            validate_pretag_package_tag_state(mutation, verifier)
        except ValueError:
            continue
        fail("pre-tag workflow tag-state mutation escaped the policy gate")
    verifier_mutations = (
        verifier.replace("local_tag_type != \"tag\"", "local_tag_type != \"commit\"", 1),
        verifier.replace(
            "validate_tag_binding(tag, commit, tag_state)",
            "pass  # tag binding skipped",
            1,
        ),
        verifier.replace(
            'choices=("absent", "annotated"), required=True',
            'choices=("absent", "annotated"), required=False',
            1,
        ),
        verifier.replace(
            "if completed.returncode == 1 and not stdout and not stderr:",
            "if completed.returncode != 0:",
            1,
        ),
        verifier.replace(
            '        git_text("cat-file", "-t", tag_ref),',
            '        "tag",',
            1,
        ),
        verifier.replace(
            '        git_text("rev-parse", f"{tag_ref}^{{commit}}").lower(),',
            "        commit.lower(),",
            1,
        ),
        verifier.replace(
            "    local_tag_sha = local_ref_sha(tag_ref)",
            "    local_tag_sha = None",
            1,
        ),
        verifier.replace(
            'if require_signature and tag_state != "annotated":',
            "if False:",
            1,
        ),
        verifier.replace(
            "    validate_signature_tag_state(args.require_signature, args.tag_state)\n"
            "    directory = args.directory.resolve()",
            "    if args.tag_state == 'annotated':\n"
            "        validate_signature_tag_state(args.require_signature, args.tag_state)\n"
            "    directory = args.directory.resolve()",
            1,
        ),
        verifier.replace(
            'license_path=extracted / "LICENSES/FFmpeg-GPL-3.0-or-later.txt",',
            'license_path=extracted / "LICENSE",',
            1,
        ),
        verifier.replace(
            'readme_path=extracted / "FFMPEG-README.txt",',
            'readme_path=extracted / "README.txt",',
            1,
        ),
        verifier.replace(
            '    if license_path is None:\n'
            '        license_path = root / "LICENSE"',
            '    license_path = root / "LICENSE"',
            1,
        ),
        verifier.replace(
            '    if readme_path is None:\n'
            '        readme_path = root / "README.txt"',
            '    readme_path = root / "README.txt"',
            1,
        ),
    )
    for mutation in verifier_mutations:
        try:
            validate_pretag_package_tag_state(release, mutation)
        except ValueError:
            continue
        fail("release-verifier tag-state mutation escaped the policy gate")


def validate_reproducible_checkout_attributes(attributes: str) -> None:
    required = [
        ".gitignore text eol=lf",
        "*.tldr text eol=lf",
        "*.dict text eol=lf",
        "*.lock text eol=lf",
        "*.ps1 text eol=lf",
        "fuzz/corpus/** -text",
    ]
    lines = attributes.splitlines()
    if any(lines.count(rule) != 1 for rule in required):
        fail("raw-hashed lockfiles and release PowerShell must be LF-stable")
    corpus_index = lines.index("fuzz/corpus/** -text")
    later_attribute_rules = [
        line
        for line in lines[corpus_index + 1 :]
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if later_attribute_rules:
        fail("opaque fuzz-corpus rule must be the final non-comment attribute rule")


def self_test_reproducible_checkout_attributes(attributes: str) -> None:
    validate_reproducible_checkout_attributes(attributes)
    for rule in [
        ".gitignore text eol=lf",
        "*.tldr text eol=lf",
        "*.dict text eol=lf",
        "*.lock text eol=lf",
        "*.ps1 text eol=lf",
        "fuzz/corpus/** -text",
    ]:
        mutation = attributes.replace(rule, "", 1)
        try:
            validate_reproducible_checkout_attributes(mutation)
        except ValueError:
            pass
        else:
            fail("checkout-attribute self-test accepted a byte-unstable release input")
    reordered = attributes.replace("fuzz/corpus/** -text\n", "", 1)
    reordered = "fuzz/corpus/** -text\n" + reordered
    try:
        validate_reproducible_checkout_attributes(reordered)
    except ValueError:
        pass
    else:
        fail("checkout-attribute self-test accepted an overridden fuzz-corpus rule")
    tabbed_override = attributes + "*.json\ttext eol=lf\n"
    try:
        validate_reproducible_checkout_attributes(tabbed_override)
    except ValueError:
        pass
    else:
        fail("checkout-attribute self-test accepted a tabbed corpus override")


def validate_versioned_release_receipts(finalizer: str) -> None:
    names = (
        "v1.7.0-improvement-audit-release-receipt.md",
        "v1.7.1-release-recovery-receipt.md",
        "v1.7.2-release-recovery-receipt.md",
        "v1.7.3-release-recovery-receipt.md",
        "v1.7.4-release-recovery-receipt.md",
        "v1.8.0-ffmpeg-9-software-baseline-receipt.md",
        "v1.8.1-patch-refresh-receipt.md",
    )
    for name in names:
        if (
            finalizer.count(f'"{name}",') != 1
            or finalizer.count(f'"docs/evidence/{name}"') != 1
        ):
            fail(f"final receipt does not bind versioned source evidence: {name}")


def self_test_versioned_release_receipts(finalizer: str) -> None:
    validate_versioned_release_receipts(finalizer)
    names = (
        "v1.7.0-improvement-audit-release-receipt.md",
        "v1.7.1-release-recovery-receipt.md",
        "v1.7.2-release-recovery-receipt.md",
        "v1.7.3-release-recovery-receipt.md",
        "v1.7.4-release-recovery-receipt.md",
        "v1.8.0-ffmpeg-9-software-baseline-receipt.md",
        "v1.8.1-patch-refresh-receipt.md",
    )
    for name in names:
        for literal in (f'"{name}",', f'"docs/evidence/{name}"'):
            mutation = finalizer.replace(literal, "", 1)
            try:
                validate_versioned_release_receipts(mutation)
            except ValueError:
                pass
            else:
                fail("versioned-receipt self-test accepted missing source evidence")


def validate_required_workflow_transport(workflow_gate: str) -> None:
    exact_contract = {
        'MAX_RECEIPT_BYTES = 128 * 1024': 1,
        'request.add_unredirected_header("Authorization", f"Bearer {token}")': 1,
        'request = github_request(url, token)': 2,
        'require_https_url(response.geturl()': 2,
        'class SecureGitHubRedirectHandler(urllib.request.HTTPRedirectHandler)': 1,
        'max_repeats = 2': 1,
        'max_redirections = 5': 1,
        'with github_opener().open(request, timeout=': 2,
        'sensitive_headers = {': 1,
        'redirected.remove_header(header)': 1,
        'redirected.add_unredirected_header("Authorization", authorization)': 1,
        'assert "Authorization" not in authenticated_request.headers': 1,
        'assert cross_origin.full_url == signed_url': 1,
        'assert cross_origin.get_header("Authorization") is None': 1,
        'assert same_origin.get_header("Authorization") == "Bearer secret"': 1,
        'assert cross_origin_back_to_api.get_header("Authorization") is None': 1,
        'ordinary_sensitive_request = urllib.request.Request(': 1,
        '"Proxy-Authorization": "Basic ordinary"': 1,
        'GitHub token was accepted for an unsafe API origin': 1,
        'required-run receipt accepted one byte beyond its limit': 1,
    }
    if (
        any(workflow_gate.count(value) != count for value, count in exact_contract.items())
        or '"Authorization": f"Bearer {token}"' in workflow_gate
    ):
        fail("required-workflow transport can leak credentials or exceed its evidence bound")
    helper = workflow_gate.split("def github_request(", 1)
    if len(helper) != 2:
        fail("required-workflow authenticated request helper is absent")
    helper_body = helper[1].split("\n\nclass ", 1)[0]
    if (
        helper_body.find('require_https_url(url, expected_host="api.github.com")') < 0
        or helper_body.find('require_https_url(url, expected_host="api.github.com")')
        > helper_body.find('request.add_unredirected_header("Authorization"')
    ):
        fail("required-workflow token can be attached before API-origin validation")


def self_test_required_workflow_transport(workflow_gate: str) -> None:
    validate_required_workflow_transport(workflow_gate)
    mutations = [
        workflow_gate.replace("add_unredirected_header", "add_header", 1),
        workflow_gate.replace("max_redirections = 5", "max_redirections = 10", 1),
        workflow_gate.replace("github_opener().open", "urllib.request.urlopen", 1),
        workflow_gate.replace("redirected.remove_header(header)", "", 1),
        workflow_gate.replace(
            'require_https_url(url, expected_host="api.github.com")', "", 1
        ),
        workflow_gate.replace(
            "MAX_RECEIPT_BYTES = 128 * 1024",
            "MAX_RECEIPT_BYTES = 64 * 1024",
            1,
        ),
    ]
    for mutation in mutations:
        try:
            validate_required_workflow_transport(mutation)
        except ValueError:
            pass
        else:
            fail("required-workflow transport mutation escaped the policy gate")


def main() -> int:
    try:
        self_test_create_only_publication_policy()
        release = (ROOT / ".github/workflows/release-trust.yml").read_text(encoding="utf-8")
        self_test_attestation_identity_policy(release)
        self_test_draft_publish_last(release)
        self_test_pinned_llvm_workflow(release)
        self_test_release_checkout_isolation_and_pretag(release)
        self_test_unequal_reproducibility_workflow(release)
        self_test_reproducible_sbom_workflow(release)
        self_test_cosign_trust_policy(release)
        sbom_policy = (ROOT / "scripts/cyclonedx_sbom.py").read_text(
            encoding="utf-8"
        )
        validate_reviewed_sbom_policy_digest(sbom_policy)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self_test_ci_gate_bodies(ci)
        adversarial = (ROOT / ".github/workflows/adversarial.yml").read_text(
            encoding="utf-8"
        )
        self_test_adversarial_patch_trigger(adversarial)
        reproducible_build = (ROOT / "scripts/build-reproducible-windows.ps1").read_text(
            encoding="utf-8"
        )
        self_test_canonical_reproducible_build(reproducible_build)
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self_test_reproducible_checkout_attributes(attributes)
        release_verifier = (ROOT / "scripts/verify-release.py").read_text(
            encoding="utf-8"
        )
        self_test_shared_sbom_verifier(release_verifier)
        self_test_pretag_package_tag_state(release, release_verifier)
        final_receipt = (ROOT / "scripts/finalize-release-receipt.py").read_text(
            encoding="utf-8"
        )
        self_test_versioned_release_receipts(final_receipt)
        workflow_gate = (ROOT / "scripts/wait-required-workflows.py").read_text(
            encoding="utf-8"
        )
        self_test_required_workflow_transport(workflow_gate)
        workflow_root = ROOT / ".github/workflows"
        workflow_paths = sorted(
            [*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")]
        )
        workflow_documents = {
            path.name: path.read_text(encoding="utf-8")
            for path in workflow_paths
        }
        self_test_pinned_workflow_actions(workflow_documents)
        workflows = "\n".join(workflow_documents.values())
        expected_tools = [
            "(?m)^cargo-auditable v0\\.7\\.5:$",
            '(cargo install --list) -join "`n"',
            'Install-PinnedCargoTool cargo-cyclonedx "cargo-cyclonedx-cyclonedx 0.5.9" cargo-cyclonedx 0.5.9 -VersionArguments @("cyclonedx", "--version")',
            'Install-PinnedCargoTool cargo-deny "cargo-deny 0.20.2" cargo-deny 0.20.2',
            'Install-PinnedCargoTool cargo-audit "cargo-audit 0.22.2" cargo-audit 0.22.2',
        ]
        auditable_reinstall = (
            "cargo install --locked --force cargo-auditable --version 0.7.5"
        )
        if (
            release.count(expected_tools[0]) != 1
            or release.count(expected_tools[1]) != 1
            or release.count(auditable_reinstall) != 1
            or release.index(auditable_reinstall) > release.index(expected_tools[1])
            or any(release.count(line) != 1 for line in expected_tools[2:])
        ):
            fail("release Cargo tools are not checked against the exact pinned binaries")
        if (
            "cargo auditable --version" in reproducible_build
            or reproducible_build.count(expected_tools[0]) != 1
            or reproducible_build.count(
                '(& rustup run 1.98.0 cargo install --list) -join "`n"'
            ) != 1
        ):
            fail("reproducible build does not probe the exact cargo-auditable package")
        ci_tools = [
            'test "$(cargo-deny --version)" = "cargo-deny 0.20.2"',
            'test "$(cargo-audit --version)" = "cargo-audit 0.22.2"',
            "cargo install --locked --force cargo-deny --version 0.20.2",
            "cargo install --locked --force cargo-audit --version 0.22.2",
        ]
        if any(ci.count(line) != 1 for line in ci_tools):
            fail("CI dependency verifiers lack exact post-install version checks")
        adversarial_tools = {
            'test "$(cargo-fuzz --version)" = "cargo-fuzz 0.13.2"': 2,
            "cargo install --locked --force cargo-fuzz --version 0.13.2": 2,
        }
        if any(adversarial.count(line) != count for line, count in adversarial_tools.items()):
            fail("adversarial verification lacks exact cargo-fuzz version checks")
        with (ROOT / "rust-toolchain.toml").open("rb") as source:
            rust = tomllib.load(source)["toolchain"]["channel"]
        if release.count(f"rustup toolchain install {rust} --profile minimal") != 1:
            fail("release Rust install does not match rust-toolchain.toml")
        with (ROOT / "policy/windows-release-license-review.toml").open("rb") as source:
            review = tomllib.load(source)
        release_pins = {
            "FFMPEG_VERSION": review["ffmpeg"]["version"],
            "FFMPEG_WINDOWS_SHA256": review["ffmpeg"]["archive_sha256"],
            "FFMPEG_WINDOWS_SIZE": review["ffmpeg"]["archive_size"],
            "FFMPEG_SOURCE_COMMIT": review["ffmpeg"]["source_commit"],
        }
        ci_pins = {
            **release_pins,
            "FFMPEG_SOURCE_SHA256": review["ffmpeg"]["source_archive_sha256"],
            "FFMPEG_SOURCE_SIZE": review["ffmpeg"]["source_archive_size"],
            "FFMPEG_SOURCE_SIGNATURE_SHA256": review["ffmpeg"]["source_signature_sha256"],
            "FFMPEG_SOURCE_SIGNATURE_SIZE": review["ffmpeg"]["source_signature_size"],
            "FFMPEG_SIGNING_KEY_SHA256": review["ffmpeg"]["signing_key_sha256"],
            "FFMPEG_SIGNING_KEY_SIZE": review["ffmpeg"]["signing_key_size"],
            "FFMPEG_SIGNING_KEY_FINGERPRINT": review["ffmpeg"]["signing_key_fingerprint"],
        }
        for name, value in release_pins.items():
            rendered = re.escape(str(value))
            if not re.search(rf'^\s*{name}:\s*"?{rendered}"?\s*$', release, re.MULTILINE):
                fail(f"release workflow {name} disagrees with checked-in review")
        for name, value in ci_pins.items():
            rendered = re.escape(str(value))
            if not re.search(rf'^\s*{name}:\s*"?{rendered}"?\s*$', ci, re.MULTILINE):
                fail(f"CI workflow {name} disagrees with checked-in review")
        if review["ffmpeg"].get("source_tag") != f'n{review["ffmpeg"]["version"]}':
            fail("checked-in FFmpeg source tag disagrees with its version")
        source_identity_fragments = (
            "Verify FFmpeg 9.0.1 source signature and identity on Unix",
            '"https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.xz.asc"',
            '"https://ffmpeg.org/ffmpeg-devel.asc"',
            'test "$(sha256_file "$archive")" = "$FFMPEG_SOURCE_SHA256"',
            'test "$(sha256_file "$signature")" = "$FFMPEG_SOURCE_SIGNATURE_SHA256"',
            'test "$(sha256_file "$signing_key")" = "$FFMPEG_SIGNING_KEY_SHA256"',
            'test "$fingerprint" = "$FFMPEG_SIGNING_KEY_FINGERPRINT"',
            'gpg --batch --verify "$signature" "$archive"',
        )
        if any(ci.count(fragment) != 1 for fragment in source_identity_fragments):
            fail("CI lacks the exact FFmpeg source hash, key, and signature gate")
        if ci.index('gpg --batch --verify "$signature" "$archive"') > ci.index('tar -xJf "$archive"'):
            fail("CI extracts FFmpeg before detached-signature verification")
        exact_ci_jobs = {
            "Linux (Ubuntu 24.04)",
            "macOS 15",
            "Windows (VS 2022)",
            "Dependency policy and supply-chain provenance",
        }
        exact_ci_steps = {
            "Check Rust formatting and JavaScript syntax",
            "Check generated capability registry",
            "Check, test, and lint on Unix",
            "Check, test, and lint on Windows",
            "Verify the vendored wgpu-hal archive and sole patch",
            "Reject stale or unowned advisory exceptions",
            "Fetch locked dependencies and verify vendored source",
        }
        if any(
            ci.count(f"name: {value}") != 1
            or workflow_gate.count(f'"{value}"') < 1
            or final_receipt.count(f'"{value}"') < 1
            for value in exact_ci_jobs | exact_ci_steps
        ):
            fail("final-candidate evidence differs from exact CI job/step names")
        if (
            "needs: verification-gate" not in release
            or "--workflow ci.yml" not in release
            or "--workflow adversarial.yml" not in release
            or "--license-reviewer" in release
            or "license-reviewed-at" in release
        ):
            fail("release publication gate or checked-in review boundary is missing")
        immutable_commit_contract = [
            "commit: ${{ steps.source.outputs.commit }}",
            '--commit "${{ steps.source.outputs.commit }}"',
            "commit: ${{ needs.verification-gate.outputs.commit }}",
            "ref: ${{ needs.reproduce-sign-publish.outputs.commit }}",
            "tag_object: ${{ steps.source.outputs.tag_object }}",
            "required_runs: ${{ steps.required.outputs.required_runs }}",
            '--target "${{ needs.verification-gate.outputs.commit }}"',
            "python scripts/wait-required-workflows.py --self-test",
            "python scripts/verify-release.py self-test",
            "python scripts/finalize-release-receipt.py self-test",
        ]
        if (
            release.count("ref: ${{ needs.verification-gate.outputs.commit }}") != 3
            or release.count("python scripts/verify-release.py annotated-tag \\") != 1
            or release.count("verify-release.py annotated-tag `") != 5
            or release.count('--tag-object "${{ needs.verification-gate.outputs.tag_object }}"') != 6
            or any(release.count(value) < 1 for value in immutable_commit_contract)
        ):
            fail("release jobs are not bound to one verified immutable commit")
        validate_create_only_publication(release)
        draft_identity_contract = [
            "draft_id: ${{ steps.initial-draft.outputs.draft_id }}",
            "draft_tag: ${{ steps.initial-draft.outputs.draft_tag }}",
            "draft_state: ${{ steps.initial-draft.outputs.draft_state }}",
            "EXPECTED_DRAFT_ID: ${{ needs.reproduce-sign-publish.outputs.draft_id }}",
            "EXPECTED_DRAFT_TAG: ${{ needs.reproduce-sign-publish.outputs.draft_tag }}",
            "EXPECTED_DRAFT_STATE: ${{ needs.reproduce-sign-publish.outputs.draft_state }}",
            '--release-database-id "$env:EXPECTED_DRAFT_ID"',
            '$releaseUrl = "https://github.com/${{ github.repository }}/releases/tag/$env:RELEASE_TAG"',
        ]
        if any(release.count(value) != 1 for value in draft_identity_contract):
            fail("draft database identity or deterministic public URL is not preserved")
        annotated_tag_contract = [
            'git_text("cat-file", "-t", tag_ref) != "tag"',
            '"ls-remote", "--tags", args.remote',
            'set(observed) != {tag_ref, peeled_ref}',
            'remote_tag_object != local_tag_object or remote_commit != args.commit',
            'args.tag_object != local_tag_object',
            '"remote_tag_row_present": True',
            '"remote_peeled_row_present": True',
        ]
        if any(release_verifier.count(value) != 1 for value in annotated_tag_contract):
            fail("annotated tag validation does not require exact remote tag and peeled rows")
        create_only_preflight_contract = [
            "def require_release_absent(",
            'method="GET"',
            "for page in range(1, 11):",
            'if release_tag == tag:',
            'f"{state} release already exists; refusing every publication mutation"',
            "release-list preflight is malformed or duplicated",
            "release-list preflight has malformed pagination",
            "release-list preflight has a noncanonical next page",
            'commands.add_parser("release-absent")',
            'assert observed_methods == ["GET"]',
            '"draft release already exists"',
            '"published release already exists"',
            "duplicate_rows = [",
            "duplicate_tag_rows = [",
            "assert page_requests == [1, 2]",
            '"HTTP 503"',
        ]
        if any(
            release_verifier.count(value) < 1
            for value in create_only_preflight_contract
        ):
            fail("create-only release preflight or its no-mutation self-test is incomplete")
        if "$null -eq $resolved" in release or "lightweight" in release.lower():
            fail("release workflow retains a lightweight-tag fallback")
        verifier_fail_closed_contract = [
            'provenance.get("schema_version") != 1',
            'provenance.get("tag") != tag',
            'provenance.get("commit") != commit.lower()',
            "validate_release_directory_inventory(directory, expected_names, args.require_signature)",
            'expected.add("SHA256SUMS.sigstore.json")',
            "entry.is_symlink() or not entry.is_file()",
            'expected_artifact_names = expected_names - {"provenance.json"}',
            "not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_names",
            "validate_provenance_artifacts(provenance, checksums, expected_names)",
            'or "\\\\" in name',
            "resolved_destination.relative_to(extraction_root)",
            "portable_name in portable_names or info.is_dir()",
        ]
        if any(
            release_verifier.count(value) != 1
            for value in verifier_fail_closed_contract
        ):
            fail("release verifier does not reject contradictory provenance or asset inventory")
        redownload = release.split("  redownload-verify:", 1)
        if len(redownload) != 2:
            fail("redownload verification job is absent")
        redownload_body = redownload[1]
        signature_index = redownload_body.find("cosign verify-blob")
        executable_verifier_index = redownload_body.find(
            "python scripts\\verify-release.py verify `"
        )
        if (
            signature_index < 0
            or executable_verifier_index < 0
            or signature_index > executable_verifier_index
        ):
            fail("redownload assets can execute before checksum signature verification")
        native_guard = r"if \(\$LASTEXITCODE -ne 0\) \{ exit \$LASTEXITCODE \}"
        if not re.search(
            rf"cosign verify-blob[\s\S]*?{native_guard}[\s\S]*?"
            rf"python scripts\\verify-release\.py verify[\s\S]*?--directory downloaded"
            rf"[\s\S]*?{native_guard}[\s\S]*?foreach \(\$asset in \$assets\)"
            rf"[\s\S]*?gh attestation verify \$asset\.FullName"
            rf"[\s\S]*?{native_guard}",
            redownload_body,
        ):
            fail("redownload trust commands do not fail immediately on native errors")
        final_receipt_order = [
            "finalize-release-receipt.py build",
            "finalize-release-receipt.py validate",
            "cosign sign-blob --yes --bundle $bundle $receipt",
            "cosign verify-blob `",
            "Attest the frozen final-release receipt",
            "gh attestation verify $receipt",
            "gh release upload \"$env:RELEASE_TAG\" $receipt $bundle",
            "gh release edit \"$env:RELEASE_TAG\" `",
            "--json databaseId,tagName,body,assets,isDraft",
            "[long]$view.databaseId -ne $expectedDraftId",
            "$view.isDraft -ne $true",
            "Compare-Object -ReferenceObject $expectedAssets -DifferenceObject $publishedAssets",
            "New-Item -ItemType Directory -Path published-complete",
            "gh release download \"$env:RELEASE_TAG\" `",
            "--dir published-complete",
            "$publishedFiles.Count -ne 12",
            "--bundle $publishedChecksumBundle",
            "--directory published-initial",
            "foreach ($asset in @(Get-ChildItem -LiteralPath published-initial -File))",
            "gh attestation verify $asset.FullName",
            "finalize-release-receipt.py validate --receipt $publishedReceipt",
            "cosign verify-blob `",
            "gh attestation verify $publishedReceipt",
            "gh release edit \"$env:RELEASE_TAG\" `",
            "--draft=false",
            "--json databaseId,tagName,url,body,assets,isDraft",
            "[long]$publishedView.databaseId -ne $expectedDraftId",
            "$publishedView.isDraft -ne $false",
            "New-Item -ItemType Directory -Path published-final-state",
            "--dir published-final-state",
            "Final published asset differs from verified draft bytes",
        ]
        cursor = 0
        for value in final_receipt_order:
            cursor = redownload_body.find(value, cursor)
            if cursor < 0:
                fail("final receipt is not built, signed, attested, uploaded, and summarized in order")
            cursor += len(value)
        post_edit = redownload_body.split('gh release edit "$env:RELEASE_TAG" `', 1)
        if len(post_edit) != 2:
            fail("release body persistence check is absent")
        if "--clobber" in redownload_body or "--pattern" in post_edit[1]:
            fail("final receipt publication can overwrite an existing release asset")
        final_receipt_contract = [
            '"receipt_kind": "collide_o_scope_external_final_release"',
            '"annotated_tag_object_sha": tag_object',
            '"peeled_commit_sha": commit',
            '"release_database_id": args.release_database_id',
            '"prepublication_state": "authenticated_draft"',
            '"required_workflows": runs',
            '"final_candidate_validation": final_validation',
            '"build_a_executable_sha256"',
            '"build_b_executable_sha256"',
            '"build_identity_sha256"',
            '"sbom_sha256"',
            '"dependency_inventory_sha256"',
            '"dependency_review_sha256"',
            '"checked_release_review_sha256"',
            '"vendor": vendor_hashes()',
            '"source_evidence_receipts": source_evidence_receipts()',
            '"assets": attested_assets',
            '"github_attestation_policy": attestation_policy',
            'github_attestation_policy(repository, tag, commit)',
            'initial.get("github_attestation_policy") != expected_attestation_policy',
            '"source_ref": f"refs/tags/{tag}"',
            '"source_digest": commit',
            '"certificate_identity": args.workflow_identity',
            '"downloaded_verification": report',
            '"status": "unavailable"',
            '"summary": summary',
            'created_and_verified_after_receipt_freeze',
            'external_fixture_ignored_count',
            'initial.get("checksum_manifest_sha256") != asset_hashes["SHA256SUMS"]',
            'sigstore.get("subject_sha256") != asset_hashes["SHA256SUMS"]',
            'package.get("sha256") != asset_hashes[package_name]',
            'version_json.get("identity_sha256") != evidence["build_identity_sha256"]',
            'ffmpeg.get("binary_sha256") != evidence["ffmpeg_binary_sha256"]',
            'shader.get("bundle_sha256") != evidence["shader_bundle_sha256"]',
            'checksum_manifest_digest(checksum_inventory)',
            'evidence[field] != checksum_inventory[name]',
            'vendor != vendor_hashes()',
            'source_receipts != source_evidence_receipts()',
            'version_json.get("version") != tag.removeprefix("v")',
            'report_ffmpeg.get("archive_sha256") != native["archive_sha256"]',
            'report_ffmpeg.get("source_commit") != native["source_commit"]',
            'report_ffmpeg.get("buildconf_sha256") != native["buildconf_sha256"]',
            'evidence.get("cargo_lock_sha256") != digest(ROOT / "Cargo.lock")',
            'evidence.get("shader_bundle_sha256") != shader_bundle_digest()',
            'package.get("source_archive_sha256")',
            'provenance_artifacts[name] != checksum_inventory[name]',
            'identities != expected_pairs',
            'observed_jobs != set(PLATFORM_TEST_STEPS)',
            'final-candidate platform test aggregates are contradictory',
            '"Arbitrary successful runner"',
            '"pe_signature_observed": False',
        ]
        if any(final_receipt.count(value) < 1 for value in final_receipt_contract):
            fail("external final-release receipt omits required release evidence")
        if (
            'if len(checksums) != 8' not in final_receipt
            or 'len(assets) != 10' not in final_receipt
            or 'observed != set(assets)' not in final_receipt
            or 'workflow in observed' not in final_receipt
            or 'copy.deepcopy(receipt)' not in final_receipt
            or '"SBOM initial asset SHA"' not in final_receipt
            or '"source evidence receipt SHA"' not in final_receipt
            or '"downloaded version/tag association"' not in final_receipt
            or '"FFmpeg source commit"' not in final_receipt
            or '"source-evidence"' not in final_receipt
        ):
            fail("final receipt inventory or mutation tests are not fail closed")
        required_run_contract = [
            '"run_id": run_id',
            '"run_attempt": attempt',
            '"url": url',
            'run.get("head_sha") != commit',
            'run.get("conclusion") != "success"',
            'args.github_output.open("a"',
            '"cargo fmt --all -- --check"',
            '"cargo check --locked --all-targets --all-features"',
            '"cargo test --locked --all-targets --all-features"',
            '"cargo clippy --locked --all-targets --all-features -- -D warnings"',
            '"dependency_exception_policy"',
            '"vendor_verifier"',
            '"external_fixture_ignored_count"',
            '2026-08-24T11:22:33.1234567Z test video::external_ffmpeg_fixture',
            'REQUIRED_WORKFLOWS = {"ci.yml", "adversarial.yml"}',
            'failed_summary_pattern = re.compile(',
            'test result: FAILED\\.',
            'exact_required_step_receipts(jobs)',
            'collect_platform_test_results(',
            'f"https://api.github.com/repos/{repository}/actions/jobs/{job_id}/logs"',
            'three summaries in one CI job satisfied three platform jobs',
            'exact CI evidence accepted required steps under a wrong job',
            'collect_final_candidate_evidence(',
            'final-candidate branch accepted an incomplete workflow set',
        ]
        if any(workflow_gate.count(value) < 1 for value in required_run_contract):
            fail("required workflow selection is not carried into the final receipt")
        if release.count("--final-candidate-evidence") != 1:
            fail("release does not collect exact final-candidate gate results and test counts")
        signing = release.split("      - name: Keyless-sign the checksum manifest", 1)
        if len(signing) != 2:
            fail("checksum signing step is absent")
        signing_body = signing[1].split("      - name: Attest release evidence", 1)[0]
        if not re.search(
            rf"cosign sign-blob[\s\S]*?{native_guard}[\s\S]*?"
            rf"cosign verify-blob[\s\S]*?{native_guard}[\s\S]*?"
            rf"python source-a\\scripts\\verify-release\.py verify"
            rf"[\s\S]*?{native_guard}",
            signing_body,
        ):
            fail("checksum signing commands do not fail immediately on native errors")
        if release.count("cosign-release: v3.1.3") != 2:
            fail("both signing and redownload jobs must pin the cosign binary")
        if (
            release.count("actions/attest-build-provenance@977bb373ede98d70efdf65b84cb5f73e068dcc2a") != 2
            or "attestations: write" not in redownload_body
            or "id-token: write" not in redownload_body
        ):
            fail("initial assets and final receipt are not both keylessly attested")
        if "verify-vendored-wgpu-hal.py --offline" in workflows:
            fail("fresh CI cannot use offline mode for the path-patched vendor archive")
        if review.get("authenticode", {}).get("status") != "unavailable":
            fail("AuthentiCode must remain an explicit unavailable stop disposition")
        if re.search(r"\b(signtool|Set-AuthenticodeSignature)\b", release, re.IGNORECASE):
            fail("workflow attempts Authenticode without a managed signing service")
    except (OSError, KeyError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"release workflow policy failed: {error}", file=sys.stderr)
        return 1
    print(
        "release workflow policy valid: annotated tag, exact pins, all-asset attestations, "
        "and signed final receipt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
