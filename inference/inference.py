import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from typing import Optional
import torch
from pathlib import Path
import json
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.model.scdnam_gpt import scDNAmGPTForSequenceClassification
from src.dataset.scdnam_dataset import TokensRatiosDataset, scDNAm_collate_TokensRatios
from sklearn import metrics
from src.mambaconfig import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from tqdm import tqdm
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn

def is_main_process():
    """
    Check if the current process is the main process in distributed training.
    
    Returns:
        bool: True if the process is the main process, False otherwise.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True  # Single-process or non-distributed mode
    return dist.get_rank() == 0

# Function to load JSON configuration files
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

# Metric computation for evaluation
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Args:
        eval_pred: Tuple of logits and labels.

    Returns:
        dict: Computed metrics (accuracy, F1 score, MCC, precision, recall).
    """
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Handle tuple logits
        logits = logits[0]

    # Compute probabilities and predictions
    probabilities = F.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(probabilities, axis=-1)

    # Filter out invalid labels
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    print("confusion_matrix", metrics.confusion_matrix(
            valid_labels, valid_predictions))

    # Compute metrics
    return {
        "accuracy": metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": metrics.f1_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "matthews_correlation": metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": metrics.precision_score(valid_labels, valid_predictions, average="macro", zero_division=0),
        "recall": metrics.recall_score(valid_labels, valid_predictions, average="macro", zero_division=0),
    }

def masked_mean(hidden_states, attention_mask):

    mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
    masked_hidden = hidden_states * mask_expanded
    valid_lengths = mask_expanded.sum(dim=1)  # [batch_size, hidden_dim]
    summed_hidden = masked_hidden.sum(dim=1)  # [batch_size, hidden_dim]
    mean_hidden = summed_hidden / valid_lengths 
    mean_hidden = torch.where(valid_lengths > 0, mean_hidden, torch.zeros_like(mean_hidden))

    return mean_hidden

class scDNAmGPTForAnalysis(scDNAmGPTForSequenceClassification):
    def __init__(
        self,
        config,
        tokenizer,
        num_labels: int,
        attention_mechanism: dict = None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        cross_attn_every_hidden_states: Optional[bool]=False,
        *args,
        **kwargs
    ) -> None:
        """Initialize the sequence classification model."""
        super().__init__(config, tokenizer, num_labels, attention_mechanism, initializer_cfg, device, dtype, need_layer_hidden_states=cross_attn_every_hidden_states, cross_attn_every_hidden_states=cross_attn_every_hidden_states)

    def forward(
        self,
        unmethy_input_ids,
        methy_input_ids,
        methy_ratios,
        labels=None,
        attention_mask=None,
        inference_params=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Perform a forward pass of the model."""

        # Get layer hidden states from the backbone
        hidden_states = self.backbone(
            unmethy_input_ids, methy_input_ids, methy_ratios,
            inference_params=inference_params,
            need_layer_hidden_states=(self.need_layer_hidden_states or self.cross_attn_every_hidden_states),
            **kwargs,
        )
        
        if self.need_layer_hidden_states or self.cross_attn_every_hidden_states:
            hidden_states, layer_hidden_states = hidden_states
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = (methy_input_ids != self.pad_token_id).int()

        if self.cross_attn_every_hidden_states:
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = layer_hidden_states[0].size()

            # Initialize list to store attention outputs for each layer
            norm_attn_outputs, attn_weights = [], []
            
            # Iterate through each layer's hidden state and apply attention separately
            layer_hidden_states = layer_hidden_states[:-2] + layer_hidden_states[-1:]
            for i, hidden_states in enumerate(layer_hidden_states):

                cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
                key_padding_mask = ~attention_mask.bool()
                # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

                if hasattr(self, "query_proj") or hasattr(self, "query_proj_layers"):
                    q_proj = self.query_proj_layers[i](cls_token_states.unsqueeze(1))
                    k_proj = self.key_proj_layers[i](hidden_states)
                    v_proj = self.value_proj_layers[i](hidden_states)
                else:
                    q_proj = cls_token_states.unsqueeze(1)
                    k_proj = hidden_states
                    v_proj = hidden_states
                
                attn_output, attn_weight = self.attention_layers[i](q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)

                # Store the attention output for this layer
                norm_attn_outputs.append(self.norm(attn_output))
                attn_weights.append(attn_weight)
            
            if return_dict:
                norm_attn_outputs_for_save = torch.cat(norm_attn_outputs, dim=1)
                
            # Aggregate the attention outputs (e.g., sum or average across layers)
            norm_attn_outputs = torch.sum(torch.cat(norm_attn_outputs, dim=1), dim=1)
            attn_weights = torch.cat(attn_weights, dim=1) #.permute(1, 0, 2)
            # print(norm_attn_outputs.shape, "norm_attn_outputs!!!!!!!!!!!!!!!!")
        
        else:
            last_non_padding_indices = attention_mask.sum(dim=1) - 1
            batch_size, seq_len, embed_dim = hidden_states.size()
            cls_token_states = hidden_states[torch.arange(batch_size), last_non_padding_indices]
            # print(cls_token_states.shape, "cls_token_states!!!!!!!!!!!!!!!!")

            key_padding_mask = ~attention_mask.bool()

            if hasattr(self, "query_proj"):
                q_proj = self.query_proj(cls_token_states.unsqueeze(1))
                k_proj = self.key_proj(hidden_states)
                v_proj = self.value_proj(hidden_states)
            else:
                q_proj = cls_token_states.unsqueeze(1)
                k_proj = hidden_states
                v_proj = hidden_states

            attn_output, attn_weights = self.attention(q_proj, k_proj, v_proj, key_padding_mask=key_padding_mask)
            attn_output = attn_output.squeeze(1)

            norm_attn_outputs = self.norm(attn_output)
            if return_dict:
                norm_attn_outputs_for_save = norm_attn_outputs
        
        # Classify using the normalized attention output
        logits = self.classify(norm_attn_outputs)

        # Return the result based on return_dict flag
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return {
            "logits": logits,
            "labels": labels,
            # "loss": loss,
            "attn_weights": attn_weights,
            "norm_attn_outputs": norm_attn_outputs_for_save,
            "hidden_states": masked_mean(hidden_states, attention_mask),
        }


# Add a function for running inference on a dataset
def run_inference_on_dataset(model, dataset, tokenizer, data_collator, device, output_dir, dataset_name, batch_size=1):
    """
    Run inference on a given dataset and save the results in structured folders and a summary CSV file.
    """
    from torch.utils.data import DataLoader
    import pandas as pd

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=8 
    )

    output_subdir = output_dir / dataset_name

    # Create specific subdirectories for each output type
    attn_weights_dir = output_subdir / "attn_weights"
    positions_dir = output_subdir / "positions"
    methy_ratios_dir = output_subdir / "methy_ratios"
    chrs_dir = output_subdir / "chrs"
    hidden_states_dir = output_subdir / "hidden_states"
    norm_attn_outputs_dir = output_subdir / "norm_attn_outputs"

    attn_weights_dir.mkdir(parents=True, exist_ok=True)
    positions_dir.mkdir(parents=True, exist_ok=True)
    methy_ratios_dir.mkdir(parents=True, exist_ok=True)
    chrs_dir.mkdir(parents=True, exist_ok=True)
    hidden_states_dir.mkdir(parents=True, exist_ok=True)
    norm_attn_outputs_dir.mkdir(parents=True, exist_ok=True)

    output_subdir.mkdir(parents=True, exist_ok=True)

    csv_path = output_subdir / f"{dataset_name}_inference_results.csv"

    if csv_path.exists():
        csv_path.unlink()
    pd.DataFrame(columns=[
        "file_name", "labels", "predictions", "chr_start", "chr_end",
        "probs", "logits", "length"
    ]).to_csv(csv_path, index=False)

    with torch.cuda.amp.autocast(), torch.no_grad():

        progress_bar = tqdm(
            loader, 
            desc="Inferencing", 
            unit="batch",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            postfix={"phase": "processing"}
        )

        for batch in progress_bar:
            # ========= forward =========
            unmethy_input_ids = batch["unmethy_input_ids"].to(device)
            methy_input_ids   = batch["methy_input_ids"].to(device)
            methy_ratios      = batch["methy_ratios"].to(device)
            positions         = batch["positions"].to(device)
            chrs              = batch["chrs"].to(device)
            labels            = batch["labels"].to(device)
            # print(len(positions[0]))

            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                unmethy_input_ids=unmethy_input_ids,
                methy_input_ids=methy_input_ids,
                methy_ratios=methy_ratios,
                attention_mask=attention_mask,
                positions=positions,
                labels=labels,
                chrs=chrs,
                file_names=batch["file_names"],
                return_dict=True
            )

            # ========= post-process =========
            predictions        = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
            probs              = F.softmax(outputs["logits"], dim=-1).cpu().numpy() #torch.sigmoid(outputs["logits"]).cpu().numpy()
            attn_weights       = outputs["attn_weights"].cpu().numpy()
            positions_cpu      = positions.cpu().numpy()     
            methy_ratios_cpu   = methy_ratios.cpu().numpy()
            chrs_cpu           = chrs.cpu().numpy()
            hidden_states      = outputs["hidden_states"].cpu().numpy()
            norm_attn_outputs  = outputs["norm_attn_outputs"].cpu().numpy()
            file_names         = batch["file_names"]

            # ========= save =========
            for i, file_name in enumerate(file_names):
                file_stub = file_name.split("/")[-1]
                
                if file_stub.endswith(".npz"):
                    np.savez(attn_weights_dir      / file_stub, data=attn_weights[i])
                    np.savez(positions_dir         / file_stub, data=positions_cpu[i])
                    np.savez(methy_ratios_dir      / file_stub, data=methy_ratios_cpu[i])
                    np.savez(chrs_dir              / file_stub, data=chrs_cpu[i])
                    np.savez(hidden_states_dir     / file_stub, data=hidden_states[i])
                    np.savez(norm_attn_outputs_dir / file_stub, data=norm_attn_outputs[i])
                else:
                    np.save(attn_weights_dir      / file_stub, attn_weights[i])
                    np.save(positions_dir         / file_stub, positions_cpu[i])
                    np.save(methy_ratios_dir      / file_stub, methy_ratios_cpu[i])
                    np.save(chrs_dir              / file_stub, chrs_cpu[i])
                    np.save(hidden_states_dir     / file_stub, hidden_states[i])
                    np.save(norm_attn_outputs_dir / file_stub, norm_attn_outputs[i])
                # print(attn_weights[i].shape[-1])

                row = {
                    "file_name": file_stub,
                    "labels":      labels[i].item(),
                    "predictions": predictions[i].item(),
                    "chr_start":   chrs_cpu[i][1].item(),
                    "chr_end":     chrs_cpu[i][-2].item(),
                    "probs":       probs[i],
                    "logits":      outputs["logits"][i].cpu().numpy(),
                    "length":      attn_weights[i].shape[-1]
                }
                # print("labels:", labels[i].item(), "predictions:", predictions[i].item())
                pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

            progress_bar.set_postfix({
                "current_samples": f"{len(file_names)}/{len(loader.dataset)}",
                "latest_file": file_names[0][:10] + "..."
            })

            del unmethy_input_ids, methy_input_ids, methy_ratios
            del labels, attention_mask
            del outputs

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def inference():
    """
    Function for inference using the scDNAm model.
    """
    parser = argparse.ArgumentParser(description="Run inference with a scDNAm model.")
    parser.add_argument("--inference_args_path", type=str, required=True, help="Path to the inference arguments JSON file.")
    args = parser.parse_args()

    # Load configurations
    inference_args_dict = load_config(args.inference_args_path)
    model_config_dict = load_config(Path(inference_args_dict["pretrained_model_path"]) / "config.json")

    # Load tokenizer
    K_mer = model_config_dict["K_mer"]
    tokenizer = AutoTokenizer.from_pretrained(f"src/tokenizers/scdnam_{K_mer}", trust_remote_code=True)

    datasets = {}

    test_csv_file = inference_args_dict.get("test_csv_file")
    if test_csv_file:
        datasets["test"] = TokensRatiosDataset(
            test_csv_file,
            inference_args_dict["test_root_path"],
            K_mer, tokenizer, inference_args_dict["type_json_path"],
            need_labels=True,
            need_analysis=True,
            max_length=inference_args_dict.get("max_length", None),
            random=inference_args_dict.get("random", False),
            use_sample=inference_args_dict.get("use_sample", False),
            use_truncation=inference_args_dict.get("use_truncation", False),
            start_idx=0,
            selective_chrs=inference_args_dict.get("selective_chrs", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        )

    val_csv_file = inference_args_dict.get("val_csv_file")
    if val_csv_file:
        datasets["validation"] = TokensRatiosDataset(
            val_csv_file,
            inference_args_dict["val_root_path"],
            K_mer, tokenizer, inference_args_dict["type_json_path"],
            need_labels=True,
            need_analysis=True,
            max_length=inference_args_dict.get("max_length", None),
            random=inference_args_dict.get("random", False),
            use_sample=inference_args_dict.get("use_sample", False),
            use_truncation=inference_args_dict.get("use_truncation", False),
            start_idx=0,
            selective_chrs=inference_args_dict.get("selective_chrs", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        )

    train_csv_file = inference_args_dict.get("train_csv_file")
    if train_csv_file:
        datasets["train"] = TokensRatiosDataset(
            train_csv_file,
            inference_args_dict["train_root_path"],
            K_mer, tokenizer, inference_args_dict["type_json_path"],
            need_labels=True,
            need_analysis=True,
            max_length=inference_args_dict.get("max_length", None),
            random=inference_args_dict.get("random", False),
            use_sample=inference_args_dict.get("use_sample", False),
            use_truncation=inference_args_dict.get("use_truncation", False),
            start_idx=0,
            selective_chrs=inference_args_dict.get("selective_chrs", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
        )

    data_collator = scDNAm_collate_TokensRatios(tokenizer=tokenizer)
    # print(datasets["train"][0])

    # Automatically choose the correct split
    split_name = 'train'  # Default split name

    # Check if 'train' exists, else fallback to 'validation' or 'test'
    if split_name not in datasets:
        if 'validation' in datasets:
            split_name = 'validation'
        elif 'test' in datasets:
            split_name = 'test'
        else:
            raise ValueError("No suitable split found in the dataset.")

    # If distributed training is being used
    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank()  # Get the rank of the current process
        device = torch.device(f"cuda:{local_rank}")  # Use the rank to assign the device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = scDNAmGPTForAnalysis.from_pretrained(
        tokenizer=tokenizer,
        pretrained_model_name=inference_args_dict["pretrained_model_path"],
        device=device,
        num_labels=datasets[split_name].num_labels,
        attention_mechanism=inference_args_dict["attention_mechanism"],
        cross_attn_every_hidden_states=inference_args_dict["cross_attn_every_hidden_states"]
    )

    # Check if Lora config exists and apply
    lora_config = inference_args_dict.get("lora_config", None)
    if lora_config:
        from peft import LoraConfig, get_peft_model, PeftModel
        peft_model_path = lora_config.get("peft_model_path", None)
        if peft_model_path:
            model = PeftModel.from_pretrained(model, peft_model_path)
            print("Loaded Lora model from pretrained path.")
        else:
            print("LoRA configuration provided, but no pretrained path specified.")
    
    # Move model to device
    model.to(device)  
    model.eval() 

    # Run inference on each dataset
    batch_size = inference_args_dict.get("batch_size", torch.cuda.device_count() * 1) 

    output_dir = Path(inference_args_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset in datasets.items():
        print(f"Running inference on the {dataset_name} dataset...")
        run_inference_on_dataset(
            model, dataset, tokenizer, data_collator, 
            device, output_dir, dataset_name, 
            batch_size=batch_size
        )
    
    if is_main_process():
        # Save inferencing arguments as JSON
        inference_args_path = os.path.join(inference_args_dict["output_dir"], "training_args.json")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(inference_args_path), exist_ok=True)
        with open(inference_args_path, "w") as f:
            json.dump(inference_args_dict, f, indent=4)
        print(f"Training arguments saved to {inference_args_path}")


if __name__ == "__main__":
    inference()
