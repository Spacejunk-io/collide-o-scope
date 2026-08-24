# Dependency license release review

`cargo deny check licenses` is the mechanical inventory gate. It proves that
the lockfile's declared SPDX expressions fit the allow-list; it is not legal
advice and does not decide distribution compatibility for the application,
FFmpeg, fonts, media, drivers, or other bundled native artifacts.

Before a release candidate is signed, the release owner must separately:

- archive `cargo deny list --format json` with the release evidence;
- review every bundled native library and asset, including FFmpeg build flags;
- verify required license texts and attribution notices are in the package;
- record the reviewer, UTC date, tag, and BuildIdentity digest in the release
  evidence; and
- reject the release if any exception is expired or lacks an owner and reason.

The checked-in `windows-release-license-review.toml` records the review owner,
UTC date, exact native archive/source/build/license hashes, notice policy, and
the explicit Authenticode-unavailable stop disposition. The release-trust
workflow enforces the mechanical parts again on the tagged source, normalizes
the inventory so a checkout path cannot enter evidence, packages
`COPYRIGHT.md` plus the pinned FFmpeg/Gyan license, source/build record, and
`-buildconf`, and binds those files and the checked review to the BuildIdentity
through signed checksums and provenance. It will not publish until the newest
CI and adversarial runs for that exact commit are green.

This checklist deliberately remains separate from `deny.toml`: the generated
record sets `legal_conclusion` to false, so a mechanical green result cannot be
mistaken for legal advice.
