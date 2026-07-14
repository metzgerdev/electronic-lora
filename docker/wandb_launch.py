"""Run the ACE-Step LoKr training with live Weights & Biases tracking.

ACE-Step's trainer logs to TensorBoard natively (no built-in W&B). We init W&B
with `sync_tensorboard=True` in THIS process, then run `train.py` in-process via
runpy so its `SummaryWriter` is mirrored to W&B live (loss, LR, grad norms). On
success the trained adapter is logged as a versioned W&B artifact.

The training argv is passed in as JSON via $TRAIN_ARGV (built by entrypoint.sh),
so this launcher owns *tracking*, not hyperparameters.

Single-GPU only: sync_tensorboard patches the SummaryWriter in this process; a
multi-process/DDP run would need per-rank handling (not used for a batch-1 LoKr).
"""
from __future__ import annotations

import json
import os
import runpy
import sys


def _config_from_argv(argv: list[str]) -> dict:
    """Pull a few flag values out of the training argv for the W&B config panel."""
    cfg: dict[str, object] = {}
    wanted = {
        "--model-variant": "variant",
        "--epochs": "epochs",
        "--learning-rate": "learning_rate",
        "--batch-size": "batch_size",
        "--gradient-accumulation": "grad_accum",
        "--lokr-linear-dim": "lokr_dim",
        "--lokr-linear-alpha": "lokr_alpha",
        "--seed": "seed",
    }
    for flag, key in wanted.items():
        if flag in argv:
            cfg[key] = argv[argv.index(flag) + 1]
    cfg["adapter"] = "lokr"
    return cfg


def main() -> int:
    workdir = os.environ.get("WORKDIR", "/workspace")
    os.chdir(workdir)  # trainer's safe_path root == cwd

    train_py = os.environ.get("TRAIN_PY", "/app/train.py")
    argv = json.loads(os.environ["TRAIN_ARGV"])  # e.g. ["train.py","--yes","fixed",...]
    output_dir = os.environ["OUTPUT_DIR"]
    variant = os.environ.get("VARIANT", "unknown")

    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "electronic-lora"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME") or f"lokr-{variant}",
        job_type="train",
        config=_config_from_argv(argv),
        sync_tensorboard=True,
        resume="allow",
    )
    print(f"[wandb] run: {run.url}", flush=True)

    # Run the trainer in-process so its TensorBoard SummaryWriter is captured.
    sys.argv = argv
    exit_code = 0
    try:
        runpy.run_path(train_py, run_name="__main__")
    except SystemExit as exc:  # argparse / trainer may sys.exit(0)
        exit_code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)

    if exit_code == 0 and os.path.isdir(output_dir):
        art = wandb.Artifact(f"lokr-{variant}", type="model")
        art.add_dir(output_dir)
        run.log_artifact(art)
        print(f"[wandb] logged adapter artifact from {output_dir}", flush=True)
    else:
        print(f"[wandb] training exit={exit_code}; artifact NOT logged", flush=True)

    run.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
