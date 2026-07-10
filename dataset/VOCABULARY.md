# Caption Vocabulary Sheet — electronic_lora_v1_core

The controlled vocabulary for all 57 captions. **Rule zero: reuse these terms
verbatim.** One concept = one exact phrase; five tracks sharing one phrase
teach one real concept, five paraphrases teach five weak ones.

## Caption anatomy (slot order)

```
[energy] [genre], [texture/flavor clauses...], [measured axis phrases...], [vocal ending]
```

Example: `driving tech house, gritty bassline pressure, dark moody tone,
weighty sub-heavy low end, sparse chopped vocal hooks`

2–7 clauses total. Short captions are fine — length variety is desirable.
Silence on an axis is intentional (mid-pack tracks say nothing on it).

## Slot 1 — energy + genre

| Energy (BPM-derived) | Genre (from tag / ear) |
|---|---|
| `driving` (≥126 BPM) | `tech house` · `house` · `bass house` · `deep house` |
| `groovy` (<126 BPM) | `funky house` · `deep tech house` · `minimal house` |

Genre = what the track IS (rhythm/tempo/structure). Influences (R&B, latin,
UK) are flavor clauses, never the genre word. Avoid word collisions with
axis phrases (prefer `deep tech house` over `minimal house` when the caption
already contains `stripped-back minimal drums`).

## Measured axes (quartile-assigned — machine wording, don't paraphrase)

Top/bottom quartile of the dataset per axis; middle 50% get no clause.

| Axis | Top quartile | Bottom quartile |
|---|---|---|
| Dynamics | `dramatic builds and drops` | `steady rolling groove` |
| Brightness | `bright crisp top end` | `dark moody tone` |
| Percussion density | `busy layered percussion` | `stripped-back minimal drums` |
| Low-end weight | `weighty sub-heavy low end` | `lean tight low end` |

**Manual refinement (by ear):** when a weighty low end is kick-dominant
rather than sub-bassline-dominant, replace with `weighty kick-driven low end`
(hyphenated). Apply consistently to every kick-dominant track, not one-off.

## Vocal ending (exactly one, last clause)

- `instrumental club track` — no vocal content at all
- `sparse chopped vocal hooks` — looped hooks/chops/spoken samples
- `chopped R&B vocal hooks` — chops whose source is audibly R&B/soul

(Full lyrical vocals are excluded from v1 by decision — no ending needed.)

## Flavor / texture clauses (ear-review vocabulary)

Standard terms already in use — reuse before inventing:

- **Bass character:** `bouncy bassline groove` · `gritty bassline pressure` ·
  `rolling bassline` · `brooding reese bassline` · `rubbery bass stabs` · `wubby fm bass`
- **Synths/leads:** `big-room synth leads` · `warm analog pads` ·
  `hypnotic minor-key arps` · `dark stabs` · `airy pads`· `vibey chords`
- **Percussion texture:** `punchy drums` · `shuffled hats` ·
  `latin percussion groove` · `dusty percussion loops`
- **Mood/energy:** `late-night energy` · `peak-time energy` ·
  `day party groove` · `hypnotic` · `brooding dark atmosphere`
- **Scene (see axis below):** `underground warehouse energy` ·
  `mainstage festival energy`
- **Influence flavors:** `chopped R&B vocal hooks` (ending slot) ·
  `soulful R&B-tinged chords` · `latin vocal chops and percussion`
- **Production:** `clean modern club mix` · `raw analog saturation`

## Scene axis (mainstream ↔ underground)

Ear-assigned gestalt axis. Use only when the **whole track** reads as one
pole; the commercial-but-credible middle (most of this dataset) stays
silent. Do NOT apply as decoration to every dark or stripped track — it
correlates with the measured axes, but it is not their synonym.

| | `underground warehouse energy` | `mainstage festival energy` |
|---|---|---|
| Arrangement | loop hypnosis; builds not telegraphed | riser + snare-roll build architecture; anthemic payoff drops |
| Mix/master | raw, dry, or saturated; dynamics intact | loud, bright, glossy sheen |
| Palette | dusty, analog, unpolished | polished sound design, big lead moments |
| Hook density | sparse; DJ-tool patience | frequent payoffs, front-loaded hooks |

Both poles are strongly pre-grounded tokens (like `R&B`) — the text encoder
already knows "warehouse" and "mainstage"; a handful of accurate examples
buys a working dial.

## Rules

1. **Verbatim reuse.** Check this sheet before writing; add here first, then use.
2. **Describe what's audible**, not provenance ("remix of X" is metadata the
   model can't hear; "R&B vocal texture over a house groove" is signal).
3. **No negations** ("no vocals" → use the instrumental ending). No filler
   ("amazing", bare "dynamic"). No artist or track names.
4. **Separation proportional to sound.** Similar tracks should share terms;
   only genuinely different tracks get different terms. A false descriptor
   bends the axis for every track that uses it.
5. **One-offs:** a unique texture clause on a single track is allowed if
   true, but weak. If a term fits ≥3 tracks, promote it to this sheet and
   apply it everywhere it's true. Current one-off tail to revisit during
   review: `quirky percussion and punchy groove`, `prowling bassline and UK
   bass energy`, `dark rolling groove with melodic hooks`, `vocal-chop hook
   and punchy drums` (contradiction on ACRAZE — resolve by ear),
   `breakbeat drums and rave stabs`, `groovy sample-led house`.
6. **Don't hand-write the trigger** — `zynarai` is prepended automatically.

## Change log

- 2026-07-09: Initial sheet extracted from the 57 live captions.
  Formalized `weighty kick-driven low end` as the third low-end value and
  the R&B flavor pair as standard terms.
- 2026-07-10: Added scene axis (`underground warehouse energy` /
  `mainstage festival energy`) with gestalt-assignment rubric; standardized
  `day party groove`. Middle of the axis stays silent by design.
