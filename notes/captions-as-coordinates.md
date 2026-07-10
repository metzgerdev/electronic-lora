# Captions Are Coordinates: What Labeling a Music Dataset Taught Me About Latent Space

*Draft material for a blog post — captured 2026-07-09 while labeling a 57-track
house dataset for an ACE-Step LoRA. Working titles: "Captions Are Coordinates",
"You're Not Describing the Track, You're Placing It", "Schema Design Is Basis
Vector Selection".*

---

## The one-line thesis

When you write training captions for a generative music model, you are not
describing tracks — you are assigning them coordinates in the model's
conditioning space. Standardized vocabulary gives you clean axes; capturing
real variance gives tracks separation along those axes; and the quality of
that geometry is the ceiling on how controllable the trained model can be.

## The discovery that started it

While auto-captioning 159 tracks with feature extraction (librosa: spectral
brightness, sub-bass ratio, onset density, RMS dynamics) mapped through
absolute thresholds, one phrase — "dynamic builds and drops" — landed on
**48 of 57** tracks in the final training set. Its counterpart phrase for the
sub-bass axis fired on **1 of 57** because the thresholds were calibrated
wrong for club masters.

A phrase that appears in ~every caption carries zero training signal. The
model can never learn what "dynamic builds and drops" *means* because it
never sees the contrast — and every house track has builds and drops anyway.
The phrase wasn't information; it was noise wearing information's clothes.

The fix: dataset-relative quantiles instead of absolute thresholds. Per
feature axis, only the top and bottom quartile of the dataset receive a
descriptor; the middle 50% stay silent on that axis. After the rewrite,
every phrase sat at exactly 15/57 — contrastive by construction.

## The geometric model

Three refinements turn "captions are coordinates" from metaphor into
something you can act on:

### 1. The space already exists — you're placing points, not building geometry

The frozen text encoder's embedding space, and the base model's learned
alignment between that space and audio, are fixed before you label anything.
"Driving", "sub-heavy", "R&B" already have locations and directions. Your
captions choose *where in the existing space* each track lands; the LoRA
learns to decode your chosen region into your sound.

Practical consequence: pre-grounded vocabulary is cheap, invented vocabulary
is expensive. "R&B" works as a control dial with five examples because the
direction already exists — you're annexing it. A made-up trigger word needs
the whole dataset behind it because you're carving meaning into a previously
meaningless point. Use standard terms everywhere except the one place you
*want* a blank token: the trigger.

### 2. Separation must be faithful — in both directions

Training aligns text-space geometry with audio-space geometry. Two failure
modes, symmetric:

- **Different sounds, same caption** → both tracks collapse to one
  conditioning point; the model treats their difference as unconditioned
  randomness. You lose control. (This is the near-constant-phrase failure.)
- **Similar sounds, different captions** → separation in text space the
  audio can't support; the model receives contradictory targets for distinct
  coordinates, and *both* dials weaken. A flattering-but-false descriptor
  doesn't just waste a token — it bends an axis for every other track that
  uses it.

So the goal is not maximum separation. It is separation in the captions
**proportional to separation in the sound**. "Accuracy beats aspiration" is
the ethical phrasing; "text-space geometry must mirror audio-space geometry"
is the mechanical one. Quantile-based labeling approximates this crudely:
extremes get named, the ambiguous middle stays silent instead of being
force-sorted.

### 3. The trigger word is the origin; descriptors are displacements

The trigger token (prepended to every caption) absorbs everything all
tracks share — genre constants, production era, the "house with builds and
drops"-ness. Caption clauses then position each track *relative to* that
shared origin. Division of labor:

- constants → the trigger (the cluster's origin)
- variance → the descriptors (coordinates within the cluster)

This is why deleting near-constant phrases was correct: a constant in the
coordinates is a bug; it belongs in the origin.

## The payoff: compositionality

If the axes are consistent (standardized vocabulary, one term per concept)
and reasonably independent (percussion density, sub weight, brightness,
vocal-source flavor), the model can recombine them at inference into
coordinates **no training track occupied**. "Stripped-back minimal drums,
weighty sub-heavy low end, chopped R&B vocal hooks" might describe zero of
the 57 tracks — but if each axis was learned cleanly, the model can navigate
there anyway.

That's the difference between a LoRA that's a style-stamp and one that's an
instrument.

## Field notes / supporting details for the post

- Vocabulary discipline in practice: pick exact phrasings ("chopped R&B
  vocal hooks", "soulful R&B-tinged chords"), write them on a sheet, reuse
  verbatim. Five tracks sharing one exact phrase teach one real concept;
  five paraphrases teach five weak ones.
- Describe what's audible, not provenance: "remix of an R&B track" is
  metadata the model can't hear; "R&B vocal texture over a house groove" is
  signal it can.
- Genre word = what the track *is* (rhythm, tempo, structure); influence
  flavors are modifier clauses. Keeps the genre axis clean.
- Fixed artist-knowledge boilerplate can contradict per-track measurement
  (one caption ended up with both "punchy drums" and "stripped-back minimal
  drums"). When knowledge and measurement disagree, measurement usually
  wins — but ears decide.
- Quartile descriptors are relative truths: "lean tight low end" means
  "leaner than 75% of *this dataset*" — on a bass-house corpus that's still
  weighty in absolute terms. Right trade for training signal; worth knowing
  at review time.
- Caption length should vary (2–7 clauses after the rewrite): a model
  trained only on essay-length captions responds poorly to short prompts.

## The bigger claim (closing section candidate)

Schema design is basis-vector selection. Deciding *which concepts get
labeled, in what vocabulary, at what granularity* is choosing the control
space a model will have — before any training run. That's the actual craft
in "data labeling", and it's why expert labeling is a different job from
transcription: the transcriber fills in a form; the expert designs the form,
and the form is the product.
