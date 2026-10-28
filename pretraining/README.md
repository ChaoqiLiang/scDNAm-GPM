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
(`pretraining_csv_file`) plus per-cell `.npz` files (token IDs, methylation
ratios, positions, chromosome IDs). The manuscript pretraining set is
`HumanbodyandMouseBrain_with_nucleotide_change_10mer_tokens_npz_repeat_one.csv`
(~950k cells: human body + mouse brain). Manifest format:

```
file_name,dimension,error
/abs/or/rel/path/to/cell.allc.npz,2165450,
```

`file_name` entries are joined with `root_path` (empty in the default config,
so the manifest carries absolute paths). Regenerate the manifest with your own
paths, or set `root_path` to a common prefix and keep file names relative.

## Run

```bash
# from the repository root (fp16, manuscript default configs)
bash pretraining/script/pretraining.sh

# or directly
torchrun --nnodes=1 --nproc_per_node=8 --master_port=16866 \
  pretraining/pretraining.py \
    --model_config_path pretraining/config/model/6mer_8mambalayers_model_config.json \
    --training_args_path pretraining/config/pretraining/training_args_fp16.json
```

The script auto-detects GPU count; `NPROC_PER_NODE`, `MODEL_CONFIG`,
`TRAINING_ARGS`, `MASTER_PORT` and `TRITON_CACHE_BASE` can be overridden via
environment variables. The entry point also maps SLURM variables
(`SLURM_PROCID` etc.) to torch.distributed, so it can be launched from an
`srun`/sbatch block as-is.

## Configs

The **manuscript pretraining setup** (defaults in `script/pretraining.sh`):

| File | Notes |
|---|---|
| `config/model/6mer_8mambalayers_model_config.json` | 6-mer, 8 Mamba layers, d_model 128, d_state 16; uses `src/tokenizers/scdnam_6mer` |
| `config/pretraining/training_args_fp16.json` | fp16, DeepSpeed, max_steps 600k, lr 1e-4; `use_dataug: false` |
| `config/pretraining/deepspeed_config_fp16.json` | ZeRO stage 0 (plain DDP-style) |

Alternatives kept for reference:

| File | Notes |
|---|---|
| `config/model/10mer_16mamba2layers_model_config_256dstate.json` | 10-mer, 16 Mamba2 layers, d_model 512; tokenizer bundled at `glm_src/tokenizers/scdnam_10mer` |
| `config/pretraining/training_args_bf16.json` | bf16 + DeepSpeed ZeRO variant |

Note on `use_dataug`: the pretraining objective is causal next-token
prediction with a methylation-ratio-weighted two-stream cross-entropy (see
`glm_src/model/scdnam_glm.py`, `scDNAmGLMLMHeadModelwithLoss.forward`). The
MLM-style masking augmentation ([MASK] replacement + ratio noise) is only
applied when `use_dataug` is `true`.

Outputs (checkpoints, logs) go to the `output_dir`/`logging_dir` given in the
training args; both are gitignored. Resume by setting `checkpoint_dir` in the
training-args JSON to an existing checkpoint directory.

## Environment

Use the same environment as fine-tuning (see repository README §2/§2b). The
Docker image `ghcr.io/chaoqiliang/scdnam-gpm` contains everything needed,
including DeepSpeed and nvcc for its JIT-compiled ops.
