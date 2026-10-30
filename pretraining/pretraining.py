import os, sys
# Standalone layout: the pretraining stack (gpm_src) lives next to this
# file, so make this directory importable instead of the repository root.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import json
import torch
import types
import argparse
import random
import numpy as np
from transformers import TrainingArguments, AutoTokenizer
from transformers import Trainer as Trainer
from gpm_src.model.scdnam_gpm import scDNAmGPMLMHeadModelwithLoss
from gpm_src.dataset.scdnam_dataset import TokensRatiosDataset, scDNAm_collate_TokensRatios
from gpm_src.mambaconfig import MambaConfig
import torch.distributed as dist
import uuid
import logging
logging.basicConfig(level=logging.DEBUG)
import warnings
warnings.simplefilter('default')  # restore the default filter so all warnings surface
warnings.filterwarnings('always', category=UserWarning)  # always show UserWarnings

def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return int(os.environ.get("RANK", "0")) == 0
    return dist.get_rank() == 0

class scDNAmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # accumulates per-micro-step statistics
        self._accum_loss_meth = 0.0
        self._accum_aux_loss = 0.0
        self._accum_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # ====== ✅ align logging with gradient_accumulation_steps ======
        if isinstance(outputs, dict):
            loss_meth = outputs.get("loss_meth", None)
            aux_loss = outputs.get("aux_loss", None)

            # 1) accumulate on every micro-step (keep floats to avoid holding the graph/VRAM)
            if loss_meth is not None:
                self._accum_loss_meth += float(loss_meth.detach())
            if aux_loss is not None:
                self._accum_aux_loss += float(aux_loss.detach())
            self._accum_count += 1

            # 2) only consider logging on the micro-step about to sync gradients (end of an optimizer step)
            if self.accelerator.sync_gradients:
                # actually log only when logging_steps fires (aligned with Trainer's step logs)
                if self.args.logging_strategy == "steps" and self.args.logging_steps > 0:
                    if (self.state.global_step % self.args.logging_steps) == 0:
                        logs = {}
                        if loss_meth is not None:
                            logs["loss_meth"] = self._accum_loss_meth / max(self._accum_count, 1)
                        if aux_loss is not None:
                            logs["aux_loss"] = self._accum_aux_loss / max(self._accum_count, 1)
                        if logs:
                            self.log(logs)

                # 3) one optimizer step finished, reset the accumulators
                self._accum_loss_meth = 0.0
                self._accum_aux_loss = 0.0
                self._accum_count = 0
        # ==========================================================

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


# Enable printing all tensor elements for debugging
def is_main_process():
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return int(os.environ.get("RANK", "0")) == 0
    return dist.get_rank() == 0

# Fix random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CUDA
    torch.backends.cudnn.benchmark = False

def load_config(json_path: str) -> dict:
    """
    Load a JSON configuration file.

    Args:
        json_path (str): Path to the JSON configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(json_path, "r") as f:
        return json.load(f)

def initialize_trainer(trainer_class, model, tokenizer, train_dataset, data_collator, training_args):
    """
    Initialize the Trainer with a custom save_model method.

    Args:
        trainer_class: Trainer class to use.
        model: The model to train.
        tokenizer: The tokenizer instance.
        train_dataset: Training dataset.
        data_collator: Data collator for batching.
        training_args: Training arguments.

    Returns:
        scDNAmTrainer: Configured trainer instance.
    """
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    def save_model(self, output_dir: str = None, **kwargs):
        """
        Override save_model to save the model and tokenizer.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)

    # Override save_model
    trainer.save_model = types.MethodType(save_model, trainer)
    return trainer

def main():
    """
    Main function for pretraining.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Pretrain a scDNAm model.")
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to the model configuration JSON file.",
    )
    parser.add_argument(
        "--training_args_path",
        type=str,
        required=True,
        help="Path to the training arguments JSON file.",
    )
    args = parser.parse_args()

    # Load model and training configurations
    model_config_dict = load_config(args.model_config_path)
    training_args_dict = load_config(args.training_args_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config_dict["tokenizer_name"], trust_remote_code=True)
    
    # Initialize model
    model_config = MambaConfig(**model_config_dict["model_config"])
    model = scDNAmGPMLMHeadModelwithLoss(
        config=model_config,
        tokenizer=tokenizer,
        initializer_cfg=model_config.initializer_cfg,
        use_dataug=training_args_dict.get("use_dataug", False)
    )

    if is_main_process():
        # Print which parameters are trainable
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"- {name}")

        # Count total and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # Initialize dataset and data collator
    pretraining_csv_file = training_args_dict["pretraining_csv_file"]
    root_path = training_args_dict["root_path"]
    K_mer = model_config_dict["model_config"]["K_mer"]
    train_dataset = TokensRatiosDataset(pretraining_csv_file, root_path, K_mer, tokenizer=tokenizer, need_labels=False,
                                        max_length=training_args_dict.get("max_length", None),
                                        use_truncation=training_args_dict.get("use_truncation", True),
                                        chrs_random=training_args_dict.get("chrs_random", True),
                                        random=training_args_dict.get("random", True),
                                        use_sample=training_args_dict.get("use_sample", False), start_idx=0)
    data_collator = scDNAm_collate_TokensRatios(tokenizer=tokenizer)

    # Initialize training arguments
    training_args = TrainingArguments(
        **training_args_dict["training_args"]
    )

    # Initialize trainer
    trainer = initialize_trainer(
        trainer_class=scDNAmTrainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        training_args=training_args,
    )
    
    # Save training arguments as JSON
    training_args_path = os.path.join(training_args_dict["training_args"]["logging_dir"], "training_args.json")
    # Ensure the directory exists
    os.makedirs(os.path.dirname(training_args_path), exist_ok=True)
    with open(training_args_path, "w") as f:
        json.dump(training_args_dict, f, indent=4)
    if is_main_process():
        print(f"Training arguments saved to {training_args_path}")

    # Resume training from checkpoint if available
    checkpoint_dir = training_args_dict.get("checkpoint_dir", None)
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        if is_main_process():
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        # Train the model
        trainer.train()

    # Save model and tokenizer
    if is_main_process():
        trainer.save_model(training_args_dict["training_args"]["output_dir"])
        tokenizer.save_pretrained(training_args_dict["training_args"]["output_dir"])


if __name__ == "__main__":
    import os, uuid, datetime
    import torch

    # ========== A) common NCCL / torch.distributed env vars to prevent hangs ==========
    # make NCCL errors surface quickly instead of silently hanging
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    # if your cluster's IB/network is flaky, try the two lines below temporarily (for debugging)
    # os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    # ========== B) fill in MASTER_ADDR/PORT only if missing (do not override torchrun) ==========
    if "MASTER_ADDR" not in os.environ:
        # scontrol only exists in Slurm environments
        hostnames = os.popen("scontrol show hostnames $SLURM_JOB_NODELIST").read().split()
        os.environ["MASTER_ADDR"] = hostnames[0] if hostnames else "127.0.0.1"

    os.environ.setdefault("MASTER_PORT", "16866")

    # ========== C) map SLURM -> torch.distributed only if missing (do not override torchrun) ==========
    if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
    if "LOCAL_RANK" not in os.environ and "SLURM_LOCALID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # read rank info (always provided by torchrun; Slurm provides it too)
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))

    # ========== D) bind GPUs ==========
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # ========== E) ✅ one Triton cache dir per rank (strongly recommended) ==========
    # otherwise concurrent triton compiles/cache writes can hang one rank -> NCCL timeout
    rank_str = str(global_rank)
    unique_id = os.environ.get("SLURM_JOB_ID", "") or str(uuid.uuid4())
    # Triton cache per rank (avoid multi-process write races -> NCCL hangs).
    # Prefer TRITON_CACHE_BASE if set; else the lab cluster path if it exists;
    # else a local directory next to this script.
    triton_base = os.environ.get("TRITON_CACHE_BASE")
    if not triton_base:
        lab_base = "/mnt/shared-storage-gpfs2/single-cell/liangchaoqi/triton_cache"
        triton_base = lab_base if os.path.isdir(lab_base) else os.path.join(HERE, "triton_cache")
    triton_cache = os.path.abspath(os.path.join(triton_base, "pretraining", unique_id, f"rank_{rank_str}"))
    os.environ["TRITON_CACHE_DIR"] = triton_cache
    os.makedirs(triton_cache, exist_ok=True)

    # ========== F) seed for reproducibility ==========
    set_seed(42)

    # ========== G) print key info ==========
    print(
        f"[rank {global_rank}/{world_size}] local_rank={local_rank} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"TRITON_CACHE_DIR={os.environ.get('TRITON_CACHE_DIR')}"
    )

    main()
