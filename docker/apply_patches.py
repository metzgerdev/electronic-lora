"""Bake the two ACE-Step-1.5 runtime patches into the image at build time.

These were applied live in every Colab session (notebook cell 3b). Here they run
once during `docker build`, against a PINNED ACE-Step commit and pinned deps, so
they are deterministic. Both are idempotent and self-verifying.

  1. vector_quantize_pytorch: `assert (<expr> > 1).all()` fires on meta-device
     tensors (newer transformers instantiates trust_remote_code models on 'meta').
     Guard each assert so it is skipped on meta tensors.
  2. acestep/training_v2/model_loader.py: force real (non-meta) weights by adding
     `low_cpu_mem_usage=False` to the AutoModel.from_pretrained call.

Run:  uv run python apply_patches.py --verify
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ACESTEP_MODEL_LOADER = "/app/acestep/training_v2/model_loader.py"

# The exact anchor the low_cpu_mem_usage=False line is inserted after. Verified
# present at the pinned ACESTEP_REF; if a bump breaks this, the build fails loudly.
ML_ANCHOR = (
    "                attn_implementation=attn_impl,\n"
    "                dtype=dtype,\n"
    "            )"
)
ML_REPLACEMENT = (
    "                attn_implementation=attn_impl,\n"
    "                dtype=dtype,\n"
    "                low_cpu_mem_usage=False,\n"
    "            )"
)


def patch_vq() -> int:
    """Guard `assert (<expr> > 1).all()` against meta-device tensors. Returns files changed."""
    import vector_quantize_pytorch  # imported from the uv env

    vq_dir = os.path.dirname(vector_quantize_pytorch.__file__)
    changed = 0
    pattern = re.compile(r"assert \(([^)]+?) > 1\)\.all\(\)")
    for fname in os.listdir(vq_dir):
        if not fname.endswith(".py"):
            continue
        fp = os.path.join(vq_dir, fname)
        with open(fp) as fh:
            text = fh.read()
        new = pattern.sub(
            r"assert \1.device.type == 'meta' or (\1 > 1).all()", text
        )
        if new != text:
            with open(fp, "w") as fh:
                fh.write(new)
            changed += 1
            print(f"  [vq] patched {fname}")
    return changed


def patch_model_loader() -> bool:
    """Insert low_cpu_mem_usage=False. Returns True if applied or already present."""
    with open(ACESTEP_MODEL_LOADER) as fh:
        text = fh.read()
    if "low_cpu_mem_usage=False" in text:
        print("  [model_loader] already patched")
        return True
    if ML_ANCHOR not in text:
        print(
            "  [model_loader] ERROR: anchor not found — the pinned ACE-Step "
            "source changed. Re-verify ACESTEP_REF and update apply_patches.py.",
            file=sys.stderr,
        )
        return False
    text = text.replace(ML_ANCHOR, ML_REPLACEMENT, 1)
    with open(ACESTEP_MODEL_LOADER, "w") as fh:
        fh.write(text)
    print("  [model_loader] patched")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Fail the build if the model_loader patch could not be applied.",
    )
    args = ap.parse_args()

    print("Applying ACE-Step runtime patches ...")
    vq_changed = patch_vq()
    ml_ok = patch_model_loader()
    print(f"vq files patched: {vq_changed}")

    if args.verify and not ml_ok:
        print("PATCH VERIFICATION FAILED", file=sys.stderr)
        return 1
    print("patches done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
