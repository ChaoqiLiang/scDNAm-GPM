# scDNAm-GPM Docker / Apptainer Environment

This directory contains everything needed to build the **exact Python/GPU
environment used in the manuscript** as a container, so that reviewers and
users can replicate results without installing `mamba-ssm`,
`causal-conv1d`, `flash-attn`, matching PyTorch/CUDA versions, etc. by hand.

The image is published automatically to GitHub Container Registry:

```
ghcr.io/chaoqiliang/scdnam-gpm:latest
```

## What is inside the image

| Component | Version |
|---|---|
| Ubuntu | 22.04 (CUDA 12.8.1 *devel* base, includes nvcc) |
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 (+ torchvision 0.22.0) |
| mamba-ssm | 2.3.0 (cu12 / torch2.7 / cp310 prebuilt wheel) |
| causal-conv1d | 1.6.0 (cu12 / torch2.7 / cp310 prebuilt wheel) |
| flash-attn | 2.8.3 (official Dao-AILab wheel) |
| transformers / DeepSpeed | 4.57.6 / 0.18.0 |
| all other pinned packages | see [`requirements-lock.txt`](requirements-lock.txt) (276 packages, exact `pip freeze` of the lab environment) |
| repository code | copied to `/workspace/scDNAm-GPM` (`PYTHONPATH` preset) |

The CUDA *devel* base is intentional: DeepSpeed JIT-compiles its ops on first
use and needs `nvcc` (`CUDA_HOME=/usr/local/cuda` is already set in the image).

**Host requirement:** an NVIDIA driver supporting CUDA 12.8 (≥ 550.54
recommended; the container brings its own CUDA runtime, only the driver must
match).

## One-time setup: host the CUDA-extension wheels (maintainers only)

`mamba-ssm 2.3.0` and `causal-conv1d 1.6.0` have no official wheels for
PyTorch 2.7, so the prebuilt wheels must be attached to a GitHub Release of
this repository; the Dockerfile downloads them from there.

1. Take the two wheels (they are also kept out of git under `docker/wheels/`
   locally, which is gitignored):

   ```
   mamba_ssm-2.3.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl   (~509 MB)
   causal_conv1d-1.6.0+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl (~275 MB)
   ```

2. On GitHub: *Releases → Draft a new release* → tag **`env-v1`** → attach
   both wheels → publish. The asset URLs must end up exactly as referenced by
   the `ARG ..._WHL` lines in [`Dockerfile`](Dockerfile) (the `+` in the file
   name is written as `%2B` in the URL).

`flash-attn` needs no upload — it is fetched from the official
[Dao-AILab release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3).

## How the image is built and published

The workflow [`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml)
builds and pushes the image on every push to `main` that changes `docker/**`,
on `v*` tags, and via manual *Run workflow*. No local Docker is needed.

To build the image yourself (e.g. for an air-gapped cluster):

```bash
docker build -t scdnam-gpm -f docker/Dockerfile .
```

## Usage — Docker

```bash
# pull (public image, no login needed)
docker pull ghcr.io/chaoqiliang/scdnam-gpm:latest

# interactive shell with GPU
docker run --gpus all -it --rm \
    -v $HOME/scDNAm-GPM-data:/workspace/scDNAm-GPM/data_and_model \
    ghcr.io/chaoqiliang/scdnam-gpm:latest bash

# run a training script directly
docker run --gpus all --rm --shm-size=16g \
    -v /path/to/data_and_model:/workspace/scDNAm-GPM/data_and_model \
    ghcr.io/chaoqiliang/scdnam-gpm:latest \
    bash script/finetuning/finetuning_colorectal_cancer_type.sh
```

Mount your `data_and_model/` (tutorial data, pretrained checkpoints) into the
container instead of rebuilding the image for every dataset.

## Usage — Apptainer / Singularity (HPC clusters)

Most HPC clusters do not allow Docker; Apptainer works without root and can
consume the same image. With the definition file in this repository:

```bash
git clone https://github.com/ChaoqiLiang/scDNAm-GPM.git
cd scDNAm-GPM/apptainer
apptainer build scDNAm-GPM.sif scDNAm-GPM.def        # or: singularity build ...
```

or in one command, straight from the registry:

```bash
apptainer build scDNAm-GPM.sif docker://ghcr.io/chaoqiliang/scdnam-gpm:latest
```

Run (always pass `--nv` so the container can see the NVIDIA driver):

```bash
# interactive shell
apptainer shell --nv scDNAm-GPM.sif

# run a script
apptainer exec --nv scDNAm-GPM.sif bash script/finetuning/finetuning_colorectal_cancer_type.sh

# GPU sanity check inside the container
apptainer exec --nv scDNAm-GPM.sif python3 -c "import torch; print(torch.cuda.is_available())"
```

Bind your data directory if it lives outside the working directory, e.g.
`--bind /your/data_and_model:/workspace/scDNAm-GPM/data_and_model`.

## Notes

- `requirements-lock.txt` is the full `pip freeze` of the lab environment minus
  internal/lab-only packages (`brainpp`, `tutel`, `refile` — the last is a
  Megvii-internal file library) that are not used by any code in this
  repository and are unavailable on PyPI, minus `torch`/`torchvision`
  which are installed separately from the CUDA-12.8 index, and minus the
  unused `scimilarity`/`tiledb*`/`zetascale` chain (zetascale requires
  scikit-learn<1.6 while umap-learn requires >=1.6, which makes the lock
  unsatisfiable; none of these packages are referenced by this repository).
  All remaining pins have been verified to exist on PyPI and to be mutually
  conflict-free via a metadata check.
- If you upgrade the wheels (new torch/CUDA combo), update the `ARG *_WHL`
  URLs in the Dockerfile and the table above, and publish a new release tag
  (e.g. `env-v2`).
