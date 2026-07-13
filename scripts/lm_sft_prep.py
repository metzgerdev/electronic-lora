"""Prepare SFT data for fine-tuning the ACE-Step 5Hz LM planner.

The 5Hz LM is a vanilla Qwen3 CausalLM that emits audio as ``<|audio_code_N|>``
tokens. To teach it your collection's composition/groove, we build supervised
examples of  (text prompt) -> (audio-code sequence)  from your tracks, then
LoRA-SFT the LM to predict the codes given the caption.

This module holds the *pure* pieces (no model needed): audio chunking, prompt
formatting (matched to the inference format seen in ACE-Step's DiT text-encoder
input logs), and jsonl writing. The model-dependent step -- turning a chunk of
audio into its ``<|audio_code_N|>`` string via
``AceStepHandler.convert_src_audio_to_codes`` -- runs in the Colab notebook,
where the base model + VAE are loaded.

Design mirrors the DiT pipeline: 57 tracks -> ~30s chunks -> ~350-450 examples,
which is far more workable for LM SFT than 57 whole-song sequences.
"""

import json
from pathlib import Path

import soundfile as sf

# The 5Hz LM prompt format, copied verbatim from ACE-Step's inference debug log
# (conditioning_text._prepare_text_conditioning_inputs). The completion (the
# audio-code string) is appended after this block during training.
PROMPT_TEMPLATE = (
    "# Instruction\n"
    "Fill the audio semantic mask based on the given conditions:\n"
    "\n"
    "# Caption\n"
    "{caption}\n"
    "\n"
    "# Metas\n"
    "- bpm: {bpm}\n"
    "- timesignature: {timesignature}\n"
    "- keyscale: {keyscale}\n"
    "- duration: {duration} seconds\n"
    "<|endoftext|>\n"
)


def build_prompt(caption, *, bpm=None, keyscale=None, timesignature=None, duration=30):
    """Render the LM text prompt for a training example.

    Empty metas are written as ``N/A`` to match how the UI serializes them
    (seen in the inference logs), so training and inference prompts line up.
    """
    return PROMPT_TEMPLATE.format(
        caption=caption.strip(),
        bpm=bpm if bpm else "N/A",
        keyscale=keyscale if keyscale else "N/A",
        timesignature=timesignature if timesignature else "N/A",
        duration=int(round(duration)),
    )


def iter_chunks(audio_path, chunk_s=30.0, min_s=8.0):
    """Yield ``(index, samples, sr, duration_s)`` for fixed-length windows.

    Trailing windows shorter than ``min_s`` are dropped (too little to learn a
    groove from, and they skew the duration meta). Stereo is preserved -- the
    encoder wants 48kHz stereo, same as the VAE.
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    frames_per_chunk = int(chunk_s * sr)
    min_frames = int(min_s * sr)

    with sf.SoundFile(str(audio_path)) as f:
        idx = 0
        while True:
            block = f.read(frames_per_chunk, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            if block.shape[0] < min_frames:
                break
            yield idx, block, sr, block.shape[0] / sr
            idx += 1


def write_chunk_wav(samples, sr, out_path):
    """Write a chunk to a wav the encoder can read; returns the path."""
    out_path = Path(out_path)
    sf.write(str(out_path), samples, sr)
    return str(out_path)


def build_record(prompt, codes_string, *, source, chunk_index):
    """One SFT example. ``completion`` is masked-in for loss; ``prompt`` masked-out."""
    return {
        "prompt": prompt,
        "completion": codes_string,
        "source": source,       # provenance: which track + chunk this came from
        "chunk_index": chunk_index,
    }


def write_jsonl(records, out_path):
    """Persist the SFT dataset; returns the count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def is_error_codes(codes_string):
    """convert_src_audio_to_codes returns a ❌-prefixed string on failure."""
    return (not codes_string) or codes_string.startswith("❌")
