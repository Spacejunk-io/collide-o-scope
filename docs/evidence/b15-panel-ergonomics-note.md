# B15 — panel ergonomics: search, filters, and per-control help

"We have more parameters than BENDR's 404 and no way to find one." This is the
first half of the plan's closing tranche: a way to find a control, and a
sentence explaining what it does once you have. The snapshot bank and the Dice
keep-masks are the second half.

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
