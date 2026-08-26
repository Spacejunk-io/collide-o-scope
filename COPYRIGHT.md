# Copyright and license boundary

This file records who owns what in this repository and under which terms it is
distributed. It is a project notice, not legal advice.

## The combined work

collide-o-scope, as assembled in this repository, is distributed under the
**GNU General Public License, version 3 or (at your option) any later
version** (`GPL-3.0-or-later`). The complete text is in [LICENSE](LICENSE).

    Copyright (C) 2026 Spacejunk-io

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along
    with this program. If not, see <https://www.gnu.org/licenses/>.

The purpose of this choice is that anyone who receives a copy of this program,
or of anything built from it, keeps the freedom to run, study, share, and
change it. Copyleft is what makes that freedom survive redistribution.

Two limits on that promise are stated here rather than left to be discovered:

- **It runs forward, not backward.** Copies already distributed under the
  fork's previous MIT grant keep those terms permanently — see point 4 below.
- **It covers conveying, not network use.** This program serves a browser
  control panel bound to `0.0.0.0:3030` and `0.0.0.0:3031` (`src/web/server.rs`).
  GPLv3 section 0 excludes "mere interaction with a user through a computer
  network with no transfer of a copy" from the definition of conveying, and
  GPLv3 imposes no network-source obligation of its own. Someone may therefore
  modify this program and operate it as a hosted or venue service without
  releasing their changes. `AGPL-3.0-or-later` is the licence that closes that
  specific hole, it is inbound-compatible with every dependency in this graph,
  and GPLv3 section 13 expressly permits combining GPLv3 and AGPLv3 code. The
  choice made here is deliberate: this is an instrument an operator runs on
  their own machine, and its panel is a local control surface rather than a
  service offered to the public. Anyone for whom the network case matters
  should treat the move to AGPL as available and unblocked.

## Upstream

This is a fork of [collide-o-scope by Luis Queral](https://github.com/luismqueral/collide-o-scope).
The original engine, compositing architecture, and effect suite are his work.

Upstream granted an MIT license in commit `a1c6b9d` ("Add MIT license",
2026-08-19):

    Copyright (c) 2026 Luis Queral

That grant is reproduced verbatim, and retained permanently, at
[LICENSES/MIT-collide-o-scope-upstream.txt](LICENSES/MIT-collide-o-scope-upstream.txt).
Retaining it is a condition of the MIT license and is not optional.

**Provenance of the grant, recorded precisely.** The MIT `LICENSE` was added to
upstream's `main` branch on 2026-08-19. This fork derives from upstream's
`feat/web-control-panel` branch (tip `1a4e100`, 2026-06-08), which carries no
`LICENSE` file of its own, and this fork's history was re-rooted at `ffa831d`,
so none of Luis's commits are ancestors of this tree. What connects them:
`feat/web-control-panel` is **fully contained** in `main` — zero commits ahead —
so every upstream commit this fork derives from is an ancestor of the commit
that carries the grant, in a repository with a single copyright holder. A
repository-root `LICENSE` at the tip of `main` is normally read as covering the
project's contents, and on that reading the grant reaches the forked tree.

That reading was initially an inference rather than a signed statement, and
this file therefore required a one-line confirmation from Luis before any
binary was published. **That confirmation has been received: on 2026-08-21,
Luis confirmed and approved the MIT grant's coverage of this fork's lineage,
including the pre-license commits and branches.** The provenance is settled,
not inferred; the reasoning above is retained as the record of why the
confirmation was worth obtaining.

## What the relicense does, and what it does not do

The MIT license is a GPL-compatible free software license. MIT-licensed code
may be combined into a GPL-licensed work, and the combined work distributed
under the GPL — MIT's grant expressly includes the right to sublicense, and
MIT's only condition, notice retention, is exactly the kind of term GPLv3
section 7(b) permits to be preserved. That is the operation performed here.

Four consequences follow, and all four are deliberate:

1. **The upstream MIT grant is untouched.** Nothing in this repository can
   revoke, narrow, or supersede it. It is a perpetual grant running to
   everyone who has or obtains a copy. Anyone may obtain
   `luismqueral/collide-o-scope` directly from upstream and use it under MIT
   terms, including in proprietary software. Even within this repository, a
   recipient who can identify the upstream-derived portions may use those
   portions under MIT: the GPL is an outbound licence that Spacejunk-io places
   on the combined work, not a retroactive change to Luis's lines. That door
   stays open, because it is not this fork's door to close.

2. **The MIT notice travels with this repository.** The upstream copyright and
   permission notice is preserved above and in `LICENSES/`, for as long as any
   upstream-derived portion remains here.

3. **The GPL governs this repository's own distribution.** Anyone who receives
   the combined fork receives it under `GPL-3.0-or-later`, and must pass on
   those same terms — including corresponding source — when they redistribute
   it or a derivative of it.

4. **Copies already released under MIT stay released under MIT.** Every commit
   up to and including `aafe671` carried an MIT grant covering this fork's own
   modifications and additions, and those commits are published. A licence
   already given cannot be taken back, so anyone may fork that tree and take it
   proprietary, permanently, and this change cannot reach them. The GPL applies
   to this and every later distribution, and to every future contribution.

A sole copyright holder may license the same work under different terms to
different recipients at different times; there is no one-licence-per-work rule.
No third-party consent was required for this change — see below.

## Contributions

Copyright in the fork's modifications and additions is held by Spacejunk-io.
Across this tree's history there are three distinct author strings but one
human: `Spacejunk-io` and `George` share the address `spacejunk572@gmail.com`,
and `Codex Release Automation <release-automation@localhost>` is this project's
own local tooling operating under the maintainer's direction, not a third-party
contribution. Luis's work arrives under MIT, which grants sublicense. No other
party holds copyright here.

By submitting a contribution to this repository you agree that it is licensed
under `GPL-3.0-or-later`.

## Derived laws and attribution

Substantial parts of the enrichment tranches (B1–B16) implement laws taken from
**BENDR** by Steve Blythe — MIT, Copyright (c) 2026 Steve Blythe. The source
files describe that relationship as "derived from" and, in several places,
"transcribed": the laws are re-expressed in this tree's own idiom (linear
light, Rec.709 luma, deterministic replay) rather than copied wholesale, but
the relationship is close enough that MIT's permission notice is reproduced in
full at [LICENSES/MIT-BENDR.txt](LICENSES/MIT-BENDR.txt) rather than merely
named. The per-site attributions throughout `src/` are part of that notice and
must not be stripped.

## Components carried in this repository

| Path | Component | Terms |
|---|---|---|
| `third_party/wgpu-hal-29.0.4/` | Vendored `wgpu-hal` 29.0.4 with a one-arm patch for [gfx-rs/wgpu#9029](https://github.com/gfx-rs/wgpu/issues/9029), marked `// LOCAL PATCH (collide-o-scope)` at `src/vulkan/swapchain/native.rs:472` | `MIT OR Apache-2.0`; **MIT is the elected option** for this copy, so the sole obligation is notice retention. See `LICENSE.MIT` and `LICENSE.APACHE` in that directory. |
| `assets/`, `static/` | Program icon, bundled panel assets | `GPL-3.0-or-later` with the combined work |

MIT and Apache-2.0 are both one-way compatible with GPLv3: their code may be
taken into a GPLv3 work. The reverse is not true, which is the point.

## Bundled typefaces — separately licensed, NOT relicensed

The `epaint_default_fonts` crate is a direct dependency (the B7 text page uses
`HACK_REGULAR` and `UBUNTU_LIGHT`; egui uses all four faces for its own
interface), and it declares `(MIT OR Apache-2.0) AND OFL-1.1 AND
Ubuntu-font-1.0` — a conjunction, not a choice. Its four TrueType files are
`include_bytes!`-embedded into the executable with no feature gate.

**Those faces keep their own licenses. They are not placed under the GPL by
being bundled here, and the GPL grant above does not extend to them.** They are
separately-licensed *data* carried alongside this program's code, each with its
own notice:

| Face | Terms | Notice |
|---|---|---|
| Hack Regular | MIT (Source Foundry Authors), incorporating DejaVu (public domain) and Bitstream Vera (Bitstream Vera License, reserved names "Bitstream" and "Vera") | [LICENSES/fonts/Hack-Regular-LICENSE.txt](LICENSES/fonts/Hack-Regular-LICENSE.txt) |
| Ubuntu Light | Ubuntu Font Licence 1.0 | [LICENSES/fonts/Ubuntu-Light-UFL-1.0.txt](LICENSES/fonts/Ubuntu-Light-UFL-1.0.txt) |
| Noto Emoji Regular | SIL Open Font License 1.1 | [LICENSES/fonts/NotoEmoji-OFL-1.1.txt](LICENSES/fonts/NotoEmoji-OFL-1.1.txt) |
| emoji-icon-font | MIT (John Slegers) | [LICENSES/fonts/emoji-icon-font-MIT.txt](LICENSES/fonts/emoji-icon-font-MIT.txt) |

Carrying those notices is what both font licenses actually require, which is
why they live here rather than only in the build cache.

**The residual tension, stated rather than waved away.** The UFL preamble
expressly permits the fonts to be "bundled, embedded, and redistributed", and
forbids only releasing *the fonts* under another licence — which is precisely
what the paragraph above declines to do. OFL 1.1 is stricter: clause 5 requires
the Font Software to be "distributed entirely under this license, and must not
be distributed under any other license", and its only carve-out is for "any
document created using the Font Software". An executable that embeds the font
bytes is a distribution of the Font Software, not a document made with it.
Under GPLv3 section 7 that clause is a further restriction matching none of the
permitted categories (a) through (f), applied to bytes inside the same binary
rather than merely aggregated with it. The position taken here — separately
licensed data, separately noticed, excluded from the GPL grant — is the
ordinary one, and is how essentially every GPL application shipping OFL fonts
operates. It is defensible; it is not formally settled.

Also worth knowing: Fedora classifies the Ubuntu Font Licence 1.0 as non-free
on drafting grounds, because it grants propagation rights without an explicit
unrestricted-use grant, and recommends the SIL OFL instead
([Fedora wiki](https://fedoraproject.org/wiki/Licensing/UbuntuFontLicense)).
That is a judgement about the font's licence, not this program's.

None of this arose under the previous MIT licence, because a permissive licence
raises no compatibility question at all. If the strictest reading is ever
wanted, the mitigation is bounded but not free: `TextPageFont::Sans` resolves at
exactly one site in `src/text_page.rs`, and egui's own font set is reachable
through epaint's optional `default_fonts` feature — but changing either alters
rastered output and breaks the text-page goldens, so it is a deliberate
decision, not a cleanup.

## External dependencies

Every release regenerates a path-free `cargo deny list --format json` inventory
from its exact `Cargo.lock`, rejects an unlicensed package, and publishes that
inventory beside the SBOM, checksums, and attributed review record. The
mechanical result is intentionally not presented as legal advice. It avoids
embedding package counts here because they become false as soon as a lockfile
changes; `dependency-license-inventory.json` in each release is the exact
machine-readable authority for that artifact.

Three details worth writing down, because each is a trap for the next audit:

- `self_cell` declares `Apache-2.0 OR GPL-2.0-only`. That is a **disjunction**,
  so Apache-2.0 is elected and the GPL-2.0-only trap is not sprung. Do not
  "simplify" that expression away when auditing.
- `winit` and `cpal` are **Apache-2.0 only**. Apache-2.0 flows into GPLv3 but
  not into GPLv2, so this work can never be offered under GPLv2. The
  compatibility being relied on runs one way, and to v3 or later only.
- The native FFmpeg libraries are **not** a Cargo dependency and are invisible
  to a `Cargo.lock` audit — `ffmpeg-sys-next` declares `links = "ffmpeg"` and
  is built without its `static` feature, so it dynamically links whatever
  FFmpeg the operator installed. See below.

Ordinary source builds do not bundle FFmpeg. The program links the FFmpeg 9
libraries through `ffmpeg-next` (the binding crates are WTFPL, which imposes
nothing, and which says nothing about the C libraries' own terms — a common
confusion) and separately invokes the `ffmpeg` and `ffprobe` command-line
tools. Those are installed by the operator; see the build instructions in
`README.md`.

The official Windows release archive is the deliberate exception. Its pinned
release-trust workflow bundles the exact checksum-verified
`Gyan.FFmpeg.Shared` 9.0.1 tools and seven ABI-major DLLs needed by the
executable. The archive
also carries the Gyan distribution's GPL text as
`LICENSES/FFmpeg-GPL-3.0-or-later.txt`, its `FFMPEG-README.txt` build/source
record, and the executable's reported `FFMPEG-BUILDCONF.txt`. Their hashes,
the upstream binary archive hash, and the runtime-reported license identity
are bound into the signed release evidence. The checked review also records the
complete Gyan external-library version inventory, including AMF
v1.5.2-2-gc35f613 and ffnvcodec n13.1.15.0-1-geddcea9. Their presence in the
distribution does not enable a Collide-o-Scope hardware path. This is a
technical provenance record, not a substitute for distribution-specific legal
review.

FFmpeg has three licence tiers, not two: `LGPL-2.1-or-later` by default,
`GPL-2.0-or-later` with `--enable-gpl`, and `GPL-3.0-or-later` with
`--enable-version3`. All three are inbound-compatible with distributing this
program under `GPL-3.0-or-later`. LGPL-2.1 section 3 permits converting a copy
to the ordinary GPL, and adds that "if a newer version than version 2 of the
ordinary GNU General Public License has appeared, then you can specify that
version instead if you wish"; a `GPL-2.0-or-later` build grants the recipient
the option of GPL-3. A `GPL-2.0-only` component would have no upgrade path and
would be fatally incompatible, but FFmpeg ships none — its GPL-only externals
(x264, x265, Xvid, frei0r, vidstab, rubberband) are all `GPL-2.0-or-later`.

The documented Windows build, `Gyan.FFmpeg.Shared` 9.0.1, self-reports
`--enable-gpl --enable-version3`, i.e. `GPL-3.0-or-later` — an exact match with
this program, requiring no version-bridging argument at all — and carries no
`--enable-nonfree`.

Two operational notes for anyone assembling a redistributable bundle rather
than relying on an operator-installed FFmpeg. First, a build configured with
`--enable-nonfree` (FDK-AAC, OpenSSL) is **not redistributable at all**,
whatever this program's licence says. Second, the command-line invocations are
mere aggregation under the FSF's own criteria — pipes, sockets, and
command-line arguments are the communication mechanisms normally used between
two separate programs — so they create no derivative work regardless of which
FFmpeg build is present.

## Warranty

There is none. See sections 15 and 16 of [LICENSE](LICENSE).
