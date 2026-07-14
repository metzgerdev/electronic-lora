# ACE-Step 1.5 LoKr training image

A CUDA Docker image that runs a headless LoKr training run on a GPU host, with
Weights & Biases tracking. The image pins ACE-Step-1.5 to a fixed commit,
installs deps from the lockfile, and applies two source fixes at build time.

## Files

| File | Role |
|---|---|
| `Dockerfile` | CUDA 12.8 / Python 3.11 image; `uv sync --frozen`; `ACESTEP_REF` pinned; patches applied at build |
| `apply_patches.py` | Applies the meta-tensor guard and the `model_loader` `low_cpu_mem_usage` fix; fails the build if the pinned source changed |
| `entrypoint.sh` | Runs: download checkpoints → preprocess → training. All state under `/workspace` |
| `wandb_launch.py` | Runs the trainer with W&B `sync_tensorboard`; logs the adapter as an artifact |
| `train.env.example` | Copy to `train.env`, set secrets and run knobs |
| `push.sh` | Build `linux/amd64` and push to a registry |

## Prerequisites

- A GPU host: an A100 for `xl_base`/`xl_sft` (4B); an L4 (24 GB) suffices for
  `base`/`sft` (2B).
- A W&B API key (`https://wandb.ai/authorize`). Without one the run logs offline.
- The `dataset/` folder (audio + `dataset.json`, ~1.6 GB). It is not in the
  image; it is mounted at run time.

## Build

Two options:

- **CI → GHCR:** `.github/workflows/build-train-image.yml` builds the image on
  push to `docker/**` (or manual dispatch) and pushes
  `ghcr.io/<owner>/electronic-lora-train:{latest,<sha>}`. Make the GHCR package
  public so hosts can pull it without credentials.
- **Local:** `./docker/push.sh <registry>/<user>/electronic-lora-train:v1`
  (build + push), or `docker build -f docker/Dockerfile -t
  electronic-lora-train:latest docker/` (build only). The build ends with
  `[model_loader] patched`; `PATCH VERIFICATION FAILED` means `ACESTEP_REF`
  drifted from the patched source.

## Configure

```bash
cp docker/train.env.example docker/train.env
# set WANDB_API_KEY, VARIANT (default xl_base), EPOCHS
```

`entrypoint.sh` selects a memory profile by variant: XL (4B) uses batch 1 /
grad-accum 4; 2B uses batch 2 / accum 2.

## Run

Mount a persistent directory at `/workspace` so downloads, tensors, and
checkpoints survive a restart; re-running skips completed stages.

Smoke test (5 epochs — also does the one-time checkpoint download and
preprocessing):

```bash
docker run --gpus all --rm \
  --env-file docker/train.env -e EPOCHS=5 \
  -v /path/to/workspace:/workspace \
  electronic-lora-train:latest
```

Full run (detached):

```bash
docker run --gpus all -d --name lokr \
  --env-file docker/train.env \
  -v /path/to/workspace:/workspace \
  electronic-lora-train:latest
docker logs -f lokr
```

Resume after an interruption: set
`RESUME_FROM=/workspace/output/lokr_<variant>_v1/checkpoints/epoch_<N>` in
`train.env`. If XL runs out of memory, add `EXTRA_TRAIN_ARGS=--offload-encoder`.

## Output

- W&B run URL is printed at start (`[wandb] run: ...`): loss, LR, gradient
  norms, and the adapter as an artifact.
- Adapter checkpoints: `/workspace/output/lokr_<variant>_v1/checkpoints/epoch_*/`.
- Offline W&B (no API key): logs in `/workspace/wandb`; upload with
  `wandb sync /workspace/wandb/offline-run-*`.

## RunPod

RunPod pulls the image; it is not built on the pod.

1. Build and push the image (CI → GHCR, or `push.sh`).
2. Create a Network Volume (~100 GB); it mounts at `/workspace`.
3. Seed the dataset onto the volume via a pod with the volume attached:
   `rsync -avP dataset/ root@<POD_IP>:/workspace/dataset/`.
4. Launch a 1× A100 pod:
   - Container Image: `ghcr.io/<owner>/electronic-lora-train:latest`
   - Network Volume at `/workspace`
   - Env: `WANDB_API_KEY`, `EPOCHS=5`, `VARIANT=xl_base`

The entrypoint runs on start and the container exits when training finishes;
read the logs / W&B, then terminate the pod. To run steps by hand, override the
pod command with `sleep infinity` and run `/opt/entrypoint.sh` in a terminal.

The same image runs on any `--gpus all` host (Vast.ai, Modal, a cloud VM).

## Notes

- `ACESTEP_REF` in the Dockerfile pins the ACE-Step commit the patches target.
  Bump it deliberately and re-run the build (the patch step re-verifies).
- Tensors and checkpoints are isolated per variant (`tensors_<variant>/`,
  `output/lokr_<variant>_v1/`), so `base`/`sft`/`xl_base`/`xl_sft` runs do not
  collide and each is a separate W&B run.
- XL (4B) does not run on a 16 GB Mac; audition adapters via the ACE-Step Gradio
  UI on a GPU host.
