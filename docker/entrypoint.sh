#!/usr/bin/env bash
# =============================================================================
# electronic-lora training entrypoint (headless, disconnect-safe).
#
# Orchestrates one training run inside the pinned image:
#   1. GPU sanity   2. resolve paths   3. download checkpoints (cached)
#   4. preprocess (cached, per-variant)   5. tracked LoKr training (W&B)
#
# Everything lives under $WORKDIR (bind-mounted persistent storage), so a killed
# box loses nothing: checkpoints, tensors and adapter output all persist and
# re-runs skip completed stages. Configure via env (see train.env.example).
# =============================================================================
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
cd "$WORKDIR"

# ---- knobs (env-overridable; defaults mirror the Colab notebook) ------------
VARIANT="${VARIANT:-xl_base}"          # base | sft | xl_base | xl_sft
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.03}"
LOKR_DIM="${LOKR_DIM:-64}"
LOKR_ALPHA="${LOKR_ALPHA:-128}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SEED="${SEED:-42}"
RESUME_FROM="${RESUME_FROM:-}"         # optional checkpoint dir to resume
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"  # e.g. "--offload-encoder"

DIT_DIR="acestep-v15-$(echo "$VARIANT" | tr '_' '-')"   # xl_base -> acestep-v15-xl-base
CKPT="${CKPT_DIR:-$WORKDIR/checkpoints}"
DATASET="${DATASET_DIR:-$WORKDIR/dataset}"
TENSORS="${TENSORS_DIR:-$WORKDIR/tensors_${VARIANT}}"
OUTPUT="${OUTPUT_DIR:-$WORKDIR/output/lokr_${VARIANT}_v1}"
mkdir -p "$CKPT" "$TENSORS" "$OUTPUT"

# XL (4B) memory profile: batch 1 / accum 4 (fits A100 w/ grad checkpointing).
# 2B: batch 2 / accum 2. Effective batch ~4 either way.
if [[ "$VARIANT" == xl_* ]]; then BATCH=1; ACCUM=4; else BATCH=2; ACCUM=2; fi

echo "==================================================================="
echo " electronic-lora training"
echo " variant=$VARIANT  epochs=$EPOCHS  lr=$LR  batch=$BATCH accum=$ACCUM"
echo " ACE-Step ref: $(cat /app/.acestep_ref 2>/dev/null || echo unknown)"
echo "-------------------------------------------------------------------"
python - <<'PY'
import torch
print("torch      :", torch.__version__)
ok = torch.cuda.is_available()
print("cuda       :", ok and torch.version.cuda or "NOT AVAILABLE (launch with --gpus all)")
if ok:
    p = torch.cuda.get_device_properties(0)
    print("gpu        :", p.name, f"{p.total_memory/1024**3:.0f} GB")
    if p.total_memory/1024**3 < 32:
        print("WARNING    : <32GB VRAM — XL (4B) may OOM; prefer an A100.")
PY
echo "==================================================================="

# ---- W&B auth (offline if no key) -------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" && "${WANDB_MODE:-}" != "disabled" ]]; then
    echo "NOTE: WANDB_API_KEY unset — setting WANDB_MODE=offline (logs locally; 'wandb sync' later)."
    export WANDB_MODE=offline
fi
export WANDB_PROJECT="${WANDB_PROJECT:-electronic-lora}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-lokr-${VARIANT}-e${EPOCHS}}"

# ---- 1. download checkpoints (cached; skip if present) ----------------------
if [[ ! -d "$CKPT/$DIT_DIR" || -z "$(ls -A "$CKPT/$DIT_DIR" 2>/dev/null)" ]]; then
    echo "[download] base snapshot + DiT variant $DIT_DIR ..."
    CKPT="$CKPT" DIT_DIR="$DIT_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download
ckpt, dit = os.environ["CKPT"], os.environ["DIT_DIR"]
snapshot_download("ACE-Step/Ace-Step1.5", local_dir=ckpt)
snapshot_download(f"ACE-Step/{dit}", local_dir=f"{ckpt}/{dit}")
print("[download] done:", dit)
PY
else
    echo "[download] $DIT_DIR present — skipping."
fi

# ---- 2. preprocess (per-variant tensors; skip if finished .pt exist) --------
FINISHED=$(find "$TENSORS" -name '*.pt' ! -name '*.tmp.pt' | wc -l | tr -d ' ')
if [[ "$FINISHED" -gt 0 ]]; then
    echo "[preprocess] $FINISHED finished tensors — skipping."
else
    echo "[preprocess] running Pass-1/2 for variant=$VARIANT ..."
    find "$TENSORS" -name '*.tmp.pt' -delete 2>/dev/null || true
    python /app/train.py fixed \
        --checkpoint-dir "$CKPT" \
        --model-variant "$VARIANT" \
        --dataset-dir "$TENSORS" \
        --output-dir "$OUTPUT" \
        --preprocess \
        --audio-dir "$DATASET/audio" \
        --dataset-json "$DATASET/dataset.json" \
        --tensor-output "$TENSORS"
    FINISHED=$(find "$TENSORS" -name '*.pt' ! -name '*.tmp.pt' | wc -l | tr -d ' ')
    echo "[preprocess] $FINISHED finished tensors."
fi

# ---- 3. tracked LoKr training -----------------------------------------------
TRAIN_ARGV=$(VARIANT="$VARIANT" CKPT="$CKPT" TENSORS="$TENSORS" OUTPUT="$OUTPUT" \
    BATCH="$BATCH" ACCUM="$ACCUM" EPOCHS="$EPOCHS" LR="$LR" \
    LOKR_DIM="$LOKR_DIM" LOKR_ALPHA="$LOKR_ALPHA" SAVE_EVERY="$SAVE_EVERY" \
    SEED="$SEED" RESUME_FROM="$RESUME_FROM" EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" \
    python3 - <<'PY'
import json, os
a = ["train.py", "--yes", "fixed",
     "--checkpoint-dir", os.environ["CKPT"],
     "--model-variant", os.environ["VARIANT"],
     "--dataset-dir", os.environ["TENSORS"],
     "--output-dir", os.environ["OUTPUT"],
     "--adapter-type", "lokr",
     "--lokr-linear-dim", os.environ["LOKR_DIM"],
     "--lokr-linear-alpha", os.environ["LOKR_ALPHA"],
     "--lokr-factor", "-1",
     "--lokr-weight-decompose",
     "--learning-rate", os.environ["LR"],
     "--epochs", os.environ["EPOCHS"],
     "--batch-size", os.environ["BATCH"],
     "--gradient-accumulation", os.environ["ACCUM"],
     "--save-every", os.environ["SAVE_EVERY"],
     "--seed", os.environ["SEED"],
     "--log-dir", os.path.join(os.environ["OUTPUT"], "runs")]
if os.environ.get("RESUME_FROM"):
    a += ["--resume-from", os.environ["RESUME_FROM"]]
extra = os.environ.get("EXTRA_TRAIN_ARGS", "").split()
a += extra
print(json.dumps(a))
PY
)
export TRAIN_ARGV VARIANT OUTPUT_DIR="$OUTPUT" WORKDIR

echo "[train] launching (W&B mode=${WANDB_MODE:-online}) ..."
echo "[train] argv: $TRAIN_ARGV"
python /opt/wandb_launch.py

echo "[done] adapter in: $OUTPUT"
echo "[done] tensorboard logs: $OUTPUT/runs"
