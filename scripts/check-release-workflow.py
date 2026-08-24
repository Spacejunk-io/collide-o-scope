#!/usr/bin/env python3
"""Static fail-closed checks for release workflow trust pins and gates."""

from __future__ import annotations

import re
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise ValueError(message)


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


def validate_reproducible_path_remapping(build_script: str) -> None:
    required = [
        "$remappedSource = $source.Replace('\\', '/')",
        "$remappedTarget = $target.Replace('\\', '/')",
        '"--remap-path-prefix=$remappedSource=/collide-o-scope"',
        '"--remap-path-prefix=$remappedTarget=/collide-o-scope-target"',
    ]
    if any(build_script.count(fragment) != 1 for fragment in required):
        fail("reproducible build must canonicalize both source and target paths exactly once")


def self_test_reproducible_path_remapping(build_script: str) -> None:
    validate_reproducible_path_remapping(build_script)
    for fragment in [
        "$remappedTarget = $target.Replace('\\', '/')",
        '"--remap-path-prefix=$remappedTarget=/collide-o-scope-target"',
    ]:
        mutation = build_script.replace(fragment, "", 1)
        try:
            validate_reproducible_path_remapping(mutation)
        except ValueError:
            pass
        else:
            fail("target-path remap self-test accepted an absolute build-path leak")


def validate_reproducible_checkout_attributes(attributes: str) -> None:
    required = [
        "*.lock text eol=lf",
        "*.ps1 text eol=lf",
    ]
    lines = attributes.splitlines()
    if any(lines.count(rule) != 1 for rule in required):
        fail("raw-hashed lockfiles and release PowerShell must be LF-stable")


def self_test_reproducible_checkout_attributes(attributes: str) -> None:
    validate_reproducible_checkout_attributes(attributes)
    for rule in ["*.lock text eol=lf", "*.ps1 text eol=lf"]:
        mutation = attributes.replace(rule, "", 1)
        try:
            validate_reproducible_checkout_attributes(mutation)
        except ValueError:
            pass
        else:
            fail("checkout-attribute self-test accepted a byte-unstable release input")


def main() -> int:
    try:
        self_test_create_only_publication_policy()
        release = (ROOT / ".github/workflows/release-trust.yml").read_text(encoding="utf-8")
        self_test_attestation_identity_policy(release)
        self_test_draft_publish_last(release)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self_test_ci_gate_bodies(ci)
        adversarial = (ROOT / ".github/workflows/adversarial.yml").read_text(
            encoding="utf-8"
        )
        reproducible_build = (ROOT / "scripts/build-reproducible-windows.ps1").read_text(
            encoding="utf-8"
        )
        self_test_reproducible_path_remapping(reproducible_build)
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self_test_reproducible_checkout_attributes(attributes)
        release_verifier = (ROOT / "scripts/verify-release.py").read_text(
            encoding="utf-8"
        )
        final_receipt = (ROOT / "scripts/finalize-release-receipt.py").read_text(
            encoding="utf-8"
        )
        workflow_gate = (ROOT / "scripts/wait-required-workflows.py").read_text(
            encoding="utf-8"
        )
        workflows = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
        )
        mutable = re.findall(r"^\s*uses:\s*([^\s#]+)", workflows, re.MULTILINE)
        if any(not re.search(r"@[0-9a-f]{40}$", value) for value in mutable):
            fail("every workflow action must use a full lowercase commit SHA")
        expected_actions = {
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
            "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830",
            "sigstore/cosign-installer@d7543c93d881b35a8faa02e8e3605f69b7a1ce62",
            "actions/attest-build-provenance@977bb373ede98d70efdf65b84cb5f73e068dcc2a",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        }
        if set(mutable) != expected_actions:
            fail("workflow action pins differ from the reviewed immutable set")
        expected_tools = [
            "(?m)^cargo-auditable v0\\.7\\.5:$",
            '(cargo install --list) -join "`n"',
            'Install-PinnedCargoTool cargo-cyclonedx "cargo-cyclonedx-cyclonedx 0.5.9" cargo-cyclonedx 0.5.9 -VersionArguments @("cyclonedx", "--version")',
            'Install-PinnedCargoTool cargo-deny "cargo-deny 0.20.2" cargo-deny 0.20.2',
            'Install-PinnedCargoTool cargo-audit "cargo-audit 0.22.2" cargo-audit 0.22.2',
        ]
        if (
            release.count(expected_tools[0]) != 2
            or release.count(expected_tools[1]) != 2
            or any(release.count(line) != 1 for line in expected_tools[2:])
        ):
            fail("release Cargo tools are not checked against the exact pinned binaries")
        if (
            "cargo auditable --version" in reproducible_build
            or reproducible_build.count(expected_tools[0]) != 1
            or reproducible_build.count(expected_tools[1]) != 1
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
        pins = {
            "FFMPEG_VERSION": review["ffmpeg"]["version"],
            "FFMPEG_WINDOWS_SHA256": review["ffmpeg"]["archive_sha256"],
            "FFMPEG_SOURCE_COMMIT": review["ffmpeg"]["source_commit"],
        }
        for name, value in pins.items():
            if not re.search(rf"^\s*{name}:\s*{re.escape(value)}\s*$", release, re.MULTILINE):
                fail(f"release workflow {name} disagrees with checked-in review")
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
            "python scripts/verify-release.py self-test",
            "python scripts/finalize-release-receipt.py self-test",
        ]
        if (
            release.count("ref: ${{ needs.verification-gate.outputs.commit }}") != 2
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
        if release.count("cosign-release: v2.6.0") != 2:
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
