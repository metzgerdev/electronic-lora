# electronic-lora

Training a rights-clean electronic/techno LoRA for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5),
end to end: dataset sourcing → expert labeling → cloud GPU training → Hugging Face release.

**Start here: [PLAN.md](PLAN.md)** — the four-phase plan with dataset spec,
per-track labeling checklist, training config, and GPU budget (~$15–50 total
on a rented RTX 4090).

## Layout

```
PLAN.md                          the plan; log at the bottom
dataset/
  audio/                         drop rights-clean tracks here (wav/flac preferred)
  dataset.template.json          ACE-Step full-format dataset JSON to copy & fill
  permissions/                   signed AI-training grants from contributing producers
```

## Ground rules

- **Provenance is the point.** Only your own tracks, tracks with written
  permission, or verified CC-licensed audio. Log the source and license for
  every file.
- **Label locally, train remotely.** All labeling/QA happens on this Mac
  (free); the NVIDIA GPU is rented only for preprocessing + training runs.
- **Never ship uncorrected auto-labels.** Whisper lyrics, BPM, and key are
  drafts until verified by ear.

## Quick reference

- Training pipeline: `ACE-Step-1.5/docs/en/LoRA_Training_Tutorial.md` (Gradio)
  and `ACE-Step-1.5/docs/sidestep/` (CLI, LoKr, VRAM profiles) in the sibling
  repo `~/Documents/Code/ACE-Step-1.5`
- Recommended config: LoKr + DoRA, rank 64, LR 0.03, ~500 epochs for ~100 tracks
- Trigger word: `zynarai` (set in dataset metadata, prepended to captions)
