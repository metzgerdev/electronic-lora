"""Prepare SFT data for fine-tuning the ACE-Step 5Hz LM planner.

The 5Hz LM is a vanilla Qwen3 CausalLM that emits audio as ``<|audio_code_N|>``
tokens. To teach it your collection's composition/groove, we build supervised
examples of  (chat prompt) -> (audio-code sequence)  from your tracks, then
LoRA-SFT the LM to predict the codes.

CRITICAL FORMAT NOTE
--------------------
The LM input is a **Qwen chat template**, NOT the ``# Instruction / # Caption /
# Metas`` block seen in the DiT text-encoder debug logs (that block is the DiT's
Qwen3-Embedding input, a different model). The real LM format, from
``llm_inference.build_formatted_prompt_with_cot``:

    apply_chat_template([
        {"role": "system", "content": "# Instruction\n{DEFAULT_LM_INSTRUCTION}\n\n"},
        {"role": "user",   "content": "# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"},
    ], add_generation_prompt=True)
    + "<think>\n\n</think>\n\n" + {codes} + <|im_end|>

So: Instruction is a SYSTEM message, user content is Caption+Lyric (no metas/
duration), and generation begins with an (empty) ``<think>`` reasoning block
before the codes. We must build this with the *actual* LM tokenizer, so the
chat-template rendering matches inference exactly — hence ``build_lm_text``
takes the tokenizer as an argument and runs in the notebook after the LM loads.

This module keeps the model-free pieces (chunking, jsonl) plus the one
tokenizer-dependent formatter, so all the format logic lives in one place.
"""

import json
from pathlib import Path

import soundfile as sf

# Verbatim from acestep/constants.py — the 5Hz LM's generation instruction.
DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"

# Empty chain-of-thought, matching the "no CoT" inference path. The doubled
# newline inside is intentional: Qwen's template renders empty reasoning as
# "<think>\n\n</think>", which is what the model saw in training.
EMPTY_COT = "<think>\n\n</think>"


def build_lm_text(tokenizer, caption, codes_string, *, lyrics="[Instrumental]",
                  instruction=DEFAULT_LM_INSTRUCTION):
    """Render one full SFT training string, matching LM inference exactly.

    Returns ``(full_text, response_marker)``. ``response_marker`` is the string
    where the completion begins (the assistant turn opener), so a
    completion-only collator can mask the prompt. Uses the real tokenizer's
    chat template so system/user wrapping is identical to inference.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {"role": "user", "content": f"# Caption\n{caption.strip()}\n\n# Lyric\n{lyrics}\n"},
        ],
        tokenize=False,
        add_generation_prompt=True,   # opens the assistant turn (e.g. <|im_start|>assistant\n)
    )
    completion = f"{EMPTY_COT}\n\n{codes_string}{tokenizer.eos_token}"
    return prompt + completion


def iter_chunks(audio_path, chunk_s=30.0, min_s=8.0):
    """Yield ``(index, samples, sr, duration_s)`` for fixed-length windows.

    Trailing windows shorter than ``min_s`` are dropped. Stereo preserved --
    the encoder wants 48kHz stereo like the VAE.
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    frames_per_chunk = int(chunk_s * sr)
    min_frames = int(min_s * sr)

    with sf.SoundFile(str(audio_path)) as f:
        idx = 0
        while True:
            block = f.read(frames_per_chunk, dtype="float32", always_2d=True)
            if block.shape[0] == 0 or block.shape[0] < min_frames:
                break
            yield idx, block, sr, block.shape[0] / sr
            idx += 1


def write_chunk_wav(samples, sr, out_path):
    """Write a chunk to a wav the encoder can read; returns the path."""
    sf.write(str(out_path), samples, sr)
    return str(out_path)


def build_record(caption, codes_string, *, source, chunk_index, lyrics="[Instrumental]"):
    """One raw SFT example. Prompt is built later with the LM tokenizer."""
    return {
        "caption": caption,
        "lyrics": lyrics,
        "codes": codes_string,
        "source": source,          # provenance: which track + chunk
        "chunk_index": chunk_index,
    }


def write_jsonl(records, out_path):
    """Persist the raw dataset; returns count written."""
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
