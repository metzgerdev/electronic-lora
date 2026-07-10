# Electronic LoRA for ACE-Step 1.5 — Project Plan

Goal: train and publish a genre LoRA (electronic/techno/house) for ACE-Step 1.5,
end-to-end: rights-clean dataset → expert labeling → cloud training → HF release.
Doubles as firsthand research for the data-labeling startup thesis.

Repo: `~/Documents/Code/ACE-Step-1.5` (MIT, active). Key docs:
- `docs/en/LoRA_Training_Tutorial.md` — official Gradio pipeline
- `docs/sidestep/` — community CLI toolkit (recommended: LoKr, VRAM profiles, estimation)

---

## Phase 1 — Dataset spec (target: 60–100 tracks)

Quality beats quantity: 50 clean, well-labeled tracks outperform 500 noisy ones.

**Sourcing (rights-clean only):**
- [ ] Your own productions / stems
- [ ] Tracks from producer friends with **written permission for AI training use**
  (one-paragraph grant; keep signed copies in `dataset/permissions/`)
- [ ] CC-BY / CC0 electronic catalogs (e.g. Free Music Archive filters) — verify
  license per track, log the URL + license in dataset notes
- [ ] NO scraped, ripped, or "found" audio. This LoRA's story is provenance.

**Audio requirements:**
- WAV or FLAC preferred (44.1 kHz); mp3/ogg/opus/m4a accepted
- Full tracks, well-mixed; avoid live rips, DJ mixes (transitions confuse
  the model), heavily mastered-clipped files
- Aim for coherent sub-style. A tight "melodic techno" LoRA beats a
  grab-bag "electronic" one. Consistency = the LoRA's identity.

**Distribution targets:**
- ~80% instrumental (mark `is_instrumental: true`), ~20% vocal OK
- BPM spread documented (e.g. 120–128); one time signature (4/4) keeps it simple

## Phase 2 — Labeling checklist (per track)

This is the expert-labeling step — do it properly, it is the whole ballgame.

| Field | How | QA rule |
|---|---|---|
| `caption` | Write by hand or draft with acestep-5Hz-lm/Gemini, then edit | Must name genre, energy, key instruments, texture ("driving melodic techno, rolling bassline, airy pads, analog arps") |
| `bpm` | Key-BPM finder or your DAW | Verify by ear against a click; half/double-time errors are common |
| `keyscale` | Same tools | Spot-check on keyboard; wrong keys teach wrong harmony |
| `timesignature` | Manual | "4" for almost all electronic |
| `lyrics` | `[Instrumental]` for instrumentals; else Whisper transcript **manually corrected**, with `[Verse]`/`[Chorus]` tags | Never ship uncorrected ASR output |
| `is_instrumental` | Manual | — |
| `custom_tag` | One trigger word for the whole set (e.g. `zynarai`) | Set once in dataset metadata, `tag_position: prepend` |

Dataset JSON: use the full ACE-Step format (see `dataset/dataset.template.json`
here, schema from `docs/sidestep/Dataset Preparation.md`). Side-Step also has a
zero-config folder mode (caption = filename, lyrics = `[Instrumental]`) — fine
for a smoke test, not for the real run.

**Suggested workflow on the Mac:** label everything locally (Gradio Dataset
Builder in ACE-Step 1.5 runs on macOS, or hand-edit the JSON), review every
track once with fresh ears, THEN rent the GPU. Never pay GPU rates to do labeling.

## Phase 3 — Training (cloud GPU)

Training is NVIDIA-centric (Flash Attention needs Ampere+; the macOS scripts
cover inference, not training). Label local, train remote.

**Recommended config (Side-Step, "Comfortable" profile on 24 GB):**
- LoKr (up to ~10× faster than plain LoRA), DoRA on, LR 0.03
- Batch 2, AdamW, rank 64; grad checkpointing + Flash Attention auto
- Epochs: ~500 for ~100 songs (800 if only 10–20)
- First: run Side-Step **estimation** (1–3 min) to rank which modules to train
- Smoke test: 10 tracks, low epochs, confirm pipeline before the full run

**GPU budget (estimates, spot prices July 2026, RunPod/Vast/Lambda class):**

| Item | Hardware | Time | Cost |
|---|---|---|---|
| Smoke test | RTX 4090 24 GB @ ~$0.40–0.70/hr | ~1 hr | < $1 |
| Preprocess 100 tracks → tensors | same | ~1–2 hr | ~$1 |
| LoKr training run | same | ~2–6 hr | $1–4 |
| Standard-LoRA comparison run (optional) | same | ~8–20 hr | $4–14 |
| Iteration ×3–4 (caption tweaks, rank, LR) | — | — | ~$10–30 |
| **Total project** | | | **~$15–50** |

(Time-per-epoch isn't published for 1.5; treat the training rows as ±2× and
recalibrate after the smoke test. An A100 80 GB @ ~$1.30–1.90/hr roughly
halves wall-clock at similar total cost.)

## Phase 4 — Evaluate & release

- [ ] A/B: base model vs LoRA on 10 fixed prompts (with and without trigger tag)
- [ ] Check for overfit: does it regurgitate training melodies? (bad) vs
      capture texture/groove? (good)
- [ ] Publish on Hugging Face: weights + dataset card listing every track's
      license/permission — the provenance story IS the differentiator
- [ ] Post in ACE-Step discussions; RapMachine and Text2Samples set the precedent
- [ ] Write up labeling-effort notes (hrs/track, error rates in auto BPM/key/ASR)
      → direct evidence for the labeling-startup thesis

## Log

- 2026-07-08: Project created. ACE-Step-1.5 cloned to ~/Documents/Code/ACE-Step-1.5.
- 2026-07-09: Triaged 161-track collection (159 keep). Auto-captioned all 159
  (librosa features + tags + artist knowledge). Trigger tag set to `zynarai`.
- 2026-07-09: v1 scope decision: instrumental-only, house/tech-house core.
  Built dataset.core.json — 57 tracks (21 instrumental, 36 with chopped
  hooks). 36 full-vocal tracks parked as v2 candidates (need transcripts).
  Note: this collection is other artists' commercial music — private
  training experiment only; the publishable LoRA still needs Phase 1
  rights-clean sourcing.
- 2026-07-10: Labeling done via Streamlit app (scripts/label_app.py +
  vocab_schema.json). All 57 signed off. Known issue accepted for v1:
  labeling pass re-diluted several phrases (underground warehouse energy
  48/57, clean modern club mix 52/57, steady rolling groove 49/57) —
  shipped as-is per pipeline-shakedown decision; fix in v2 (add per-pill
  frequency counters to the app). Dataset staged (audio/ hardlinks +
  dataset.json). Training: Colab Pro L4 via notebooks/colab_train_lokr.ipynb
  (LoKr dim64/alpha128 + DoRA, LR 0.03, 100 epochs per Side-Step guidance
  for 50+ songs, batch 2, checkpoints -> Drive every 10 epochs).
