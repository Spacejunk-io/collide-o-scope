# B15 — panel ergonomics, the snapshot bank, and Dice keep-masks

"We have more parameters than BENDR's 404 and no way to find one." This is the
plan's closing tranche, in four parts: a way to find a control, a sentence
explaining what it does, eight whole-rig slots to travel between, and a way to
throw the dice without losing the part you liked.

## One help table, two consumers

The panel needs help text to search over and to show; the native patch
parameter editor needs the same sentences as hover tooltips. Two hand-kept
copies would drift the first time a law changed, so the table lives in
`src/control_help.rs` and the browser's copy is **generated** from it by
`control_help::panel_javascript` and served as `help.js` — the shared-parse-table
law the wire vocabularies already follow, applied to prose. The native editor
hovers the same entries through `help_for_any`, keyed by the same wire
parameter name its rows already carry. One description per control, on both
surfaces, by construction.

**187 entries**, covering every `data-param`, `data-temporal`, and `data-ntsc`
row the panel shows. House voice: what the control does, and *why it behaves
the way it does* — the second half being the part that is hard to recover from
the code, and therefore the part worth writing down. Where a law is documented
in CLAUDE.md the text states it exactly; where behaviour is a matter of degree
it stays general rather than inventing a precision the engine does not promise.

Coverage is proven against the shipped markup rather than asserted.
`every_panel_control_row_has_help_and_no_entry_is_an_orphan` fails both when a
visible control has no sentence and when an entry describes a control that no
longer exists, so a future tranche that adds a row without writing its sentence
fails here. `every_entry_is_a_usable_sentence` keeps a floor under the prose —
it caught two stub entries that were labels rather than help. The generated
asset escapes `<`, `>`, and `&`, so no entry can close the script tag or inject
markup.

## Search and filters cost the engine nothing

`/` focuses a search over control name, section, and help text. **MOVING**
narrows to controls a modulation route currently drives; **CHANGED** narrows to
controls sitting away from their default.

All three are a view over data the panel already holds: **no new wire action,
no round trip, nothing asked of the render thread.** The test asserts the
feature's whole block contains neither `sendAction(` nor `ws.send`, so a future
edit that reaches for the engine has to make that choice deliberately. A
filtered-out control is *hidden*, never disabled — a route driving a hidden
control keeps driving it, and clearing the query restores every row untouched.

**Cost discipline.** The index walks the DOM only when the filter criteria
change, or on a snapshot while a filter is actually engaged. With nothing
filtered, the 30 Hz state packet does no extra work at all.

**Matching is split in two, deliberately.** Identity text — label, section,
parameter name — takes a plain substring, so `phos` finds Persistence Red.
Help prose takes a **word-start** match, because a substring over sentences is
noise: `gain` was matching *a-gain-st*. One regex is compiled per pass, not per
row.

**MOVING** derives from the compiled route table already in the snapshot. The
engine's target naming is transcribed as three exact-equality rules rather than
a two-hundred-entry map:

```text
target == param                          master effects, melt_*, mosh_*, sync_*
target == "temporal_" + param            feedback, slit_*, fb_*, loom_*, atlas_*, garden_*
target == "display_" + param[5..]        disp_*
```

Because every rule is an exact string equality, a target the rules cannot place
lights **nothing** rather than lighting the wrong row.

**CHANGED** needed a default per control, and the panel already had them —
three tables inlined inside the binder functions. They are hoisted to module
scope and now serve both the double-click reset and the filter, so the two
cannot disagree about what "default" means. No table was copied. A checkbox's
authored default is what the markup ships checked, exactly as a select's is its
`selected` option. Where the panel genuinely does not know a default it reports
*not changed* rather than guessing: a filter that invents differences is worse
than one with an honest blind spot, and that blind spot is the families with no
defaults table (transform, motion, gyro, pad).

## Ten reset bugs, found by writing the guard

CHANGED needed to know each control's default, which is what exposed the fact
that the panel's own reset path often did not.

The binder resolves a double-click reset as `defaults[param] ?? min`, so any
control absent from that table resets to its slider **minimum**. That is
correct for the many controls whose default *is* their minimum, and silently
wrong for every bipolar control, whose default is zero while its minimum is
negative. Ten were shipping:

| control | reset actually sent | should be |
|---|---|---|
| `fb_rotate` | −5 deg/tick | 0 |
| `fb_offset_x` | −0.5 | 0 |
| `fb_offset_y` | −0.5 | 0 |
| `fb_hue_rotate` | −180 deg | 0 |
| `slit_angle` | −180 deg | 0 |
| `loom_phase` | **−1000** | 0 |
| `loom_angle` | −180 deg | 0 |
| `sync_bias` | −1 | 0 |
| `head_switching_shift` | −100 | 0 |
| `composite_sharpening` | −1 | 0 |

`every_panel_default_agrees_between_the_markup_and_the_reset_table` closes the
class rather than the instances: the panel states each default twice — once in
the value span the operator first sees, once in the reset table a double-click
actually sends — and the test compares those two sources for every range row in
all three families, reporting every disagreement in one run.

One honest note on that test: its first parser was line-based, and the master
defaults table packs several entries per line, so it reported **fifteen false
disagreements**. They were checked rather than "fixed"; with the parser
corrected, only the two genuine NTSC cases remained. A guard that finds
problems is only useful if its findings are verified before they are acted on.

## Proof

Hosted: the seven `control_help` fixtures (uniqueness, the prose floor,
scope-and-bare lookup, generated-table well-formedness and escaping, hostile
escaping, panel-row coverage with orphan rejection, and load order), the search
contract test (`sendAction`/`ws.send` absence, the three transcribed rules, the
idle-cost line, hidden-never-disabled, the slash shortcut yielding to any field
being typed in, the tooltip source), the served-asset test, and the
defaults-agreement test across all three families.

Driven against the running app, which is the only way to test JavaScript this
tree has no runner for:

- 187 help tooltips present at load, before any search;
- `phos` → the four phosphor controls by identity; `p22` → two of them by
  **help text alone**; `gain` → five real gain controls with the *against*
  false match gone;
- a no-match query hides all 274 rows and empties their groups; clearing
  restores all 274 and every group;
- CHANGED is **empty** on a pristine program and exactly `["Pixelate"]` after
  editing pixelate;
- MOVING is empty with no routes, and all six mapping rules resolve while both
  refusals (wrong row, unknown family) hold.

Full six-step gate green at the final tree state: **1,597 passed, 0 failed**.

No Rust render path changed, so no exactness re-measurement applies: the
tranche adds no pass, no uniform, and no shader.

## Pins

`index.html` range count stays **202** — the search input is `type="search"`,
and the filters are buttons. `app.js` literal template tags stay **24**.
`GENERATOR_VERSION` stays "12", the sidecar schema stays 6, and the renderer
texture floor stays 30.

## The snapshot bank

Eight whole-rig slots and one glide time. The spec left one decision to
implementation — widen Morph, or build a bank that recalls *through* it — and
the second is the one taken, for the reason the spec itself gives: **recall
does not invent a second way to interpolate a rig.** A slot holds exactly what
a Morph slot holds; a recall captures the live rig into A, loads the slot into
B, and starts a glide. Ownership transfer, midpoint discretes, wrapped hue
arcs, and stale-topology purges are therefore the laws that already exist,
and there is only ever one answer to "what lies between two rigs".

The bank owns storage and nothing else:

- **Fixed width.** Eight slots, never growable — a bank is a row of buttons an
  operator learns by position, and a growable one would make slot 5 mean
  something different tomorrow. An out-of-range slot is refused rather than
  clamped onto a neighbour, because a button that silently wrote elsewhere
  would be worse than one that did nothing.
- **Barriered like a capture.** Save and recall both carry the two revision
  barriers `morph_capture` carries, are ordering barriers in both queues, and
  are purged from the queue by any topology edit — they capture the same thing,
  so they inherit the same hazard.
- **Empty means empty.** Recalling an empty slot is refused rather than
  recalling a default rig.
- **Carried whole in patches**, skip-serialized when untouched, so every
  pre-B15 patch keeps its bytes and its canonical hash. A short, long, or
  hostile bank sanitizes to the fixed width with a non-finite glide taking the
  neutral default rather than an extreme.

## Dice keep-masks

`keep_source`, `keep_modulation`, and `keep_output_chain` on the existing Dice
action, each defaulting to false — the established behaviour — so an unflagged
throw is byte-identical to every throw before them.

They compose safely because every Dice draw already runs in its own stable,
domain-separated stream: the master draws on stream 0 keyed by the master seed,
each layer on stream `index + 1` keyed by its own. Skipping a domain therefore
**cannot shift what another domain draws**, and the fixture proves exactly
that: with the source kept, the master's draw is byte-identical to the
unflagged throw; with the output chain kept, the layers' draws are.

The domains map onto real code paths rather than onto new bookkeeping — the
master chain (effects, transform, rack, composition, motion, temporal
originals), the layers (effects, transform, rack, motion), and the modulation
matrix (the LFO seeds). One documented consequence: the master seed is part of
the output chain, so a throw that keeps it does not advance the program's dice
cursor, which makes a modulation-only throw a pure function of the current
seed. That corner is documented rather than papered over; the useful cases —
keep my sources, keep my modulation — are unaffected.

Adding a layer needs a renderer, which hosted tests do not have, so the layer
domain is proven by auditing its four guards rather than by executing them.
The master and modulation domains are executed.

## Live verification of the second half

Driven against the running app: eight slots render; shift-clicking slot 3
stores the rig and the label becomes `[3]` with the status reading "1 of 8
slots stored"; moving the program and clicking the slot recalls it, and the
Morph status reads **"gliding to B — 2.02 beats remaining"** with the fader
mid-travel at 0.49 — the recall demonstrably travels through the existing
Morph pair rather than snapping; alt-clicking empties the slot and the status
returns to its prompt.

## Final pins

`index.html` range count **203** (the bank's recall glide is the one addition
beyond the search half's 202). `app.js` literal template tags stay **24**.
`GENERATOR_VERSION` stays "12" — the keep-masks are live Dice, not the
generator. The sidecar schema stays 6 and the renderer texture floor stays 30.
The tranche adds no pass, no uniform, and no shader.

Full six-step gate green at the final tree state: **1,602 passed, 0 failed**.
