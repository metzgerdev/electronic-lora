# Professional training run — Lambda + pinned Docker + W&B

A reproducible, headless replacement for the Colab notebook. Instead of
re-hacking the environment every session, you build a **pinned image once**
(ACE-Step at a fixed commit, deps from the lockfile, the two runtime patches
baked in), run it **headless on a Lambda A100**, and watch it live in **W&B**.

```
build image (once)  ──►  Lambda A100 box  ──►  docker run --env-file train.env
                                                  │
                              W&B (loss/LR/artifact)   /workspace (persistent)
```

## What's here

| File | Role |
|---|---|
| `Dockerfile` | Pinned CUDA 12.8 / Py3.11 image; `uv sync --frozen`; patches baked at build |
| `apply_patches.py` | Bakes the meta-tensor + `model_loader` fixes (was notebook cell 3b) |
| `entrypoint.sh` | Orchestrates download → preprocess → tracked training; all under `/workspace` |
| `wandb_launch.py` | Mirrors the trainer's TensorBoard to W&B live + logs the adapter as an artifact |
| `train.env.example` | Copy to `train.env`, fill in secrets/knobs |

## Prerequisites

- A [Lambda Cloud](https://lambda.ai) account and an SSH key added to it.
- A [W&B](https://wandb.ai) account + API key (`https://wandb.ai/authorize`).
- Your `dataset/` (audio + `dataset.json`) locally — it is **not** in the image.

## 1. Launch a Lambda instance

From the Lambda dashboard (or `lambda-cloud` CLI): launch a **1× A100 (40 or
80 GB)** instance, attach a **persistent filesystem** (so checkpoints/tensors
survive a teardown), and note the instance IP. Then:

```bash
ssh ubuntu@<INSTANCE_IP>
```

Lambda images ship NVIDIA drivers + Docker + the nvidia-container-toolkit, so
`--gpus all` works out of the box.

## 2. Put code + data on the box

From your Mac:

```bash
# the docker/ folder (build context) + your dataset
rsync -avP docker/                ubuntu@<IP>:~/electronic-lora/docker/
rsync -avP dataset/               ubuntu@<IP>:~/workspace/dataset/      # ~2.5 GB
```

`~/workspace` is the persistent dir we bind-mount into the container.

## 3. Build the image (on the box)

```bash
cd ~/electronic-lora
docker build -f docker/Dockerfile -t electronic-lora-train:latest docker/
```

First build is slow (clone + `uv sync` + patch verify). It ends with
`[model_loader] patched` / `already patched` — if it prints `PATCH
VERIFICATION FAILED`, the pinned `ACESTEP_REF` drifted; re-verify before training.

## 4. Configure the run

```bash
cp docker/train.env.example docker/train.env
nano docker/train.env      # set WANDB_API_KEY; VARIANT=xl_base; EPOCHS=5 for the smoke test
```

## 5. Smoke test first (measure per-epoch time, prove the pipeline)

```bash
docker run --gpus all --rm \
  --env-file docker/train.env \
  -e EPOCHS=5 \
  -v ~/workspace:/workspace \
  electronic-lora-train:latest
```

This does the one-time checkpoint download (~25 GB+) and preprocessing (cached
to `/workspace`), then 5 epochs. Confirm in W&B that loss is logged and drops,
and read the per-epoch wall-clock to project the full-run cost. If XL OOMs, add
`EXTRA_TRAIN_ARGS=--offload-encoder` to `train.env`.

## 6. Full run (headless, survives disconnect)

```bash
docker run --gpus all -d --name lokr \
  --env-file docker/train.env \
  -v ~/workspace:/workspace \
  electronic-lora-train:latest
docker logs -f lokr          # detach any time; the run continues
```

Because downloads, tensors, and checkpoints all live in `/workspace` (persistent
filesystem), a killed box or container loses nothing — re-run the same command
and it skips completed stages. To resume mid-training after a crash, set
`RESUME_FROM=/workspace/output/lokr_xl_base_v1/checkpoints/epoch_<N>` in `train.env`.

## 7. Monitor & retrieve

- **Live:** the W&B run URL is printed at start (`[wandb] run: ...`) — loss, LR,
  gradient norms, and the final adapter as a versioned **artifact**.
- **Adapter:** `~/workspace/output/lokr_<variant>_v1/checkpoints/epoch_*/`.
  Pull it back with `rsync -avP ubuntu@<IP>:~/workspace/output/ ./output/`, or
  `wandb artifact get` from anywhere.
- **Offline W&B** (no key at run time): logs sit in `/workspace/wandb`; upload
  later with `wandb sync ~/workspace/wandb/offline-run-*`.

## 8. Tear down (stop paying)

```bash
docker rm -f lokr
```

Then **terminate the instance** from the Lambda dashboard. The persistent
filesystem keeps your checkpoints for the next session; the GPU meter stops.

## Notes

- **Pinning:** the image is fixed to `ACESTEP_REF` in the Dockerfile (the commit
  the patches were verified against). Bump it deliberately and re-run the build's
  patch verification — do not float to `main`.
- **Eval:** XL is a 4B decoder and won't run on the 16 GB Mac; audition via the
  ACE-Step Gradio UI on a GPU box (the notebook's last cell, or a separate run).
- **Multi-run A/B** (`xl_base` vs `xl_sft`, caption tweaks): change `VARIANT` /
  knobs in `train.env` and re-run — tensors and outputs are isolated per variant,
  and each shows up as its own W&B run for side-by-side comparison.
