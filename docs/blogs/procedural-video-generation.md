<div class="mw6 center tl mb4">

### procedural video generation

the render pipeline is done. you load a patch — layers, blend modes, effects, VHS degradation — and it spits out an mp4 at whatever resolution you want. headless wgpu, no window needed, ffmpeg on the other end catching raw RGBA frames.

now the question becomes: what if you don't write the patches by hand?

---

### the shape of the problem

collide-o-scope already has a parameter space. every knob in the web panel maps to a float or a bool in a YAML file. a patch is just a point in that space — layer selection, blend modes, effect intensities, VHS settings. rendering is deterministic given a patch.

so generation reduces to: how do you navigate a high-dimensional parameter space and land on things that look good?

---

### random walks

the strategy I keep coming back to is brownian motion through patch space. start from a known-good configuration — something you made live, something you saved because it felt right — and perturb it.

a temperature knob controls how far each step drifts:

- 0.1 — barely perceptible. same vibe, slightly different grain.
- 0.5 — recognizable family resemblance. the interesting zone.
- 1.0 — same ballpark but you wouldn't have made this by hand.
- 2.0 — might be ugly. might be the best thing you've rendered all week.

this assumes a corpus of starting patches. which means the first real feature is frictionless patch collection — a button in the web panel that snapshots current state to a `patches/` folder without interrupting the session. perform for ten minutes, end up with thirty anchor points.

---

### no categorization needed

the library is just a folder of video files. no tags, no metadata, no manual sorting. and that's fine.

two options:

treat all clips as undifferentiated material. any clip layers with any other clip. the blend mode and opacity are the composition logic. accidental juxtaposition is how a lot of video art actually gets made.

or — infer lightweight properties at scan time. we already decode frames for thumbnails. computing average luminance, dominant color, and motion energy (frame-diff RMS) is almost free. enough to say "don't stack three dark clips" or "pair one static texture with one high-motion source." no tagging UI. just heuristics from pixel data.

I'd start with pure random and add the inferred properties when outputs feel too samey.

---

### titles from markov chains

every generated piece needs a name. not "untitled_003" — something that feels like it belongs to the work.

the corpus: CRT and VHS terminology. words like tracking, composite, chroma, drift, snow, smear, blanking, interlace, field, sync, phase, sweep, decay, signal, ghost, dropout, phosphor, luminance, raster. maybe 100–150 terms organized as a bigram adjacency map.

a second-order chain trained on two-word phrases from that vocabulary produces:

- composite drift
- phase decay
- phosphor ghost
- signal blanking
- chroma sweep

weight the chain toward vocabulary that relates to the active params in the patch. heavy VHS settings pull from analog words. digital effects pull from a different register — pixel, grid, quantize, step, threshold.

no ML. a JSON file and thirty lines of rust.

---

### sonification — the part I can't stop thinking about

what if the video and audio are the same data expressed in two modalities?

not audio-reactive video (that's been done to death). the inverse: video-reactive audio. the visual parameters determine how a source audio file gets processed.

take a sample — a drone, a field recording, a synth pad — and run it through effects that mirror the visual chain:

| visual | audio |
|--------|-------|
| pixelate | bitcrusher |
| grain | noise floor |
| VHS tracking | wow & flutter (pitch wobble) |
| snow | white noise blend |
| luma_smear | reverb / convolution blur |
| hue_shift | pitch shift |
| contrast | compression |
| vignette | low-pass filter |
| breathe | tremolo |
| rgb_split | stereo delay spread |

the source provides musical content. the patch shapes the texture. a clean sine wave through a high-VHS patch sounds like a warped tape. the same patch with a piano sample sounds like a half-remembered recording. the mapping creates coherence without manual sync.

implementation: process audio sample-by-sample (or in small blocks per frame), write PCM alongside video, mux with ffmpeg at the end. no DAW, no cpal dependency for offline rendering. just math on floats.

---

### the closed system

this is what makes the whole thing cohere. one PatchState produces:

1. a video (rendered through the GPU + NTSC pipeline)
2. an audio track (source material processed through mapped effects)
3. a title (markov chain weighted by active parameters)
4. metadata (seed, lineage, extracted stats)

every output is internally consistent by construction. the title sounds like the video looks like the audio feels. no separate composition step. no manual alignment.

and because everything derives from a serializable YAML file, the entire history is reproducible. given the same seed, library, and source audio, you get the same piece. or you bump the temperature and get its sibling.

---

### what to build first

1. **patch collection** — quick-save button in web panel. frictionless. builds the corpus.
2. **random walk engine** — takes a patch + temperature, produces N variations. CLI tool.
3. **markov titles** — curated word list, bigram model, weighted by params. tiny.
4. **audio processing** — load a WAV, apply mapped effects per-frame, write output. proof of concept with one sample file.
5. **batch generator** — ties 1–4 together. `--generate --count 10 --temperature 0.5 --audio drone.wav`

each step is useful on its own. they compose into something larger.

---

### open questions

- should random walks respect constraints (e.g. "tracking noise without head switching looks bad") or should we trust curation after the fact?
- is frame-by-frame audio processing the right granularity, or should param changes map to longer envelopes?
- what's the right source audio? single long drone? stems? does the generator pick audio too?
- how do you browse/curate 100 generated pieces efficiently? thumbnail grid? auto-playlists?

these are decisions, not blockers. the pipeline exists. now it's about what to feed it.

</div>
