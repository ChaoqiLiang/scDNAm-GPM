# Pretraining

Standalone pretraining code for scDNAm-GPM (masked methylation language
modeling on scWGBS data, Mamba backbone). It is deliberately **isolated from
the rest of the repository**: nothing under `src/` (used by fine-tuning and
inference) is imported or modified, so adding this folder cannot break the
existing pipeline.

## Layout

```
pretraining/
├── pretraining.py            # entry point (HF Trainer, torchrun/SLURM-aware)
├── glm_src/                  # self-contained pretraining stack
│   ├── model/scdnam_glm.py               # scDNAmGLMLMHeadModelwithLoss
│   ├── model/scdnam_mixer_seq_simple.py  # Mamba backbone
│   ├── dataset/scdnam_dataset.py         # dataset + collator (chrs_random etc.)
│   ├── mambaconfig.py
│   └── tokenizers/scdnam_10mer/          # 10-mer tokenizer (paper pretrain config)
├── config/
│   ├── model/…json           # model configs (tokenizer_name + model_config)
│   └── pretraining/…json     # training args + DeepSpeed config
└── script/pretraining.sh     # torchrun launcher
```

`glm_src` is the newer pretraining stack from the internal scDNAm_GLM project.
The only edits relative to the internal version: internal imports rewritten
from `src.*` to `glm_src.*`, the `sys.path` bootstrap points at this folder,
and the per-rank Triton cache path falls back to a local directory when the
lab-cluster path does not exist (override with `TRITON_CACHE_BASE`).

## Data requirements

Training reads what `data_process/process_scwgbs.py` produces: a CSV manifest
(`pretraining_csv_file`) plus per-cell `.npz` files under `root_path`
(token IDs, methylation ratios, positions, chromosome IDs). Point both fields
in the training-args JSON at your data.

## Run

```bash
# from the repository root
bash pretraining/script/pretraining.sh

# or directly
torchrun --nnodes=1 --nproc_per_node=8 --master_port=16866 \
  pretraining/pretraining.py \
    --model_config_path pretraining/config/model/6mer_16mamba2layers_model_config_64dstate.json \
    --training_args_path pretraining/config/pretraining/training_args_bf16.json
```

The script auto-detects GPU count; `NPROC_PER_NODE`, `MODEL_CONFIG`,
`TRAINING_ARGS`, `MASTER_PORT` and `TRITON_CACHE_BASE` can be overridden via
environment variables. The entry point also maps SLURM variables
(`SLURM_PROCID` etc.) to torch.distributed, so it can be launched from an
`srun`/sbatch block as-is.

## Configs

| File | Notes |
|---|---|
| `config/model/6mer_16mamba2layers_model_config_64dstate.json` | 6-mer variant; uses the tokenizer already bundled at `src/tokenizers/scdnam_6mer` |
| `config/model/10mer_16mamba2layers_model_config_256dstate.json` | 10-mer variant (the manuscript pretraining setup); uses `glm_src/tokenizers/scdnam_10mer` |
| `config/pretraining/training_args_bf16.json` | bf16 + DeepSpeed ZeRO; edit `pretraining_csv_file`, `root_path` and the `output_dir`/`logging_dir` paths |
| `config/pretraining/deepspeed_config_bf16.json` | ZeRO-2 config consumed via the `deepspeed` field of the training args |

Outputs (checkpoints, logs) go to the `output_dir`/`logging_dir` given in the
training args; both are gitignored. Resume by setting `checkpoint_dir` in the
training-args JSON to an existing checkpoint directory.

## Environment

Use the same environment as fine-tuning (see repository README §2/§2b). The
Docker image `ghcr.io/chaoqiliang/scdnam-gpm` contains everything needed,
including DeepSpeed and nvcc for its JIT-compiled ops.
