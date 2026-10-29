import gzip
import numpy as np
import pandas as pd
import torch, json, random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Sequence
from dataclasses import dataclass


# Define a global K-mer configuration dictionary
K_mer_dict = {
    "8mer": {"N_mod": 8192, "seq_len": 425000},
    "6mer": {"N_mod": 512, "seq_len": 2000000},
    "4mer": {"N_mod": 32, "seq_len": 2000000},
    "2mer": {"N_mod": 2, "seq_len": 2000000},
}

# hg38 chromosome lengths
chr_len_dict = {
    '1': 248956422, '10': 133797422, '11': 135086622, '12': 133275309,
    '13': 114364328, '14': 107043718, '15': 101991189, '16': 90338345,
    '17': 83257441, '18': 80373285, '19': 58617616, '2': 242193529,
    '20': 64444167, '21': 46709983, '22': 50818468, '3': 198295559,
    '4': 190214555, '5': 181538259, '6': 170805979, '7': 159345973,
    '8': 145138636, '9': 138394717,
    -2: 16569,        # M
    -1: 156040895,    # X
    0: 57227415,      # Y
}

def normalize_chr_id(c):
    """
    Normalize chromosome id to a key usable in chr_len_dict.
    Supports int, str like '1', 'chr1', 'X', 'chrX', etc.
    """
    if isinstance(c, bytes):
        c = c.decode()

    # int-like values
    if isinstance(c, (np.integer, int)):
        if str(c) in chr_len_dict:
            return str(c)
        if c in chr_len_dict:
            return c  # for -2, -1, 0
        return str(c)

    s = str(c)
    s = s.replace("chr", "").replace("CHR", "").strip()
    if s.upper() == "X":
        return -1
    if s.upper() == "Y":
        return 0
    if s.upper() in ("M", "MT"):
        return -2

    if s in chr_len_dict:
        return s
    try:
        si = int(s)
        if str(si) in chr_len_dict:
            return str(si)
        if si in chr_len_dict:
            return si
    except Exception:
        pass
    return s

# class TokensRatiosDataset(Dataset):
#     """
#     A PyTorch Dataset for handling tokenized sequences and methylation ratios.

#     Args:
#         csv_file (str): Path to the CSV file containing file information.
#         root_path (str): Root directory where data files are stored.
#         K_mer (str): The K-mer type, must be a key in `K_mer_dict`.
#     """
#     def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
#                  type_json_path: str = None, batch_type_json_path: str = None,
#                  need_labels: bool = False, need_batch: bool = False,
#                  need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, min_length: int = 50000, max_length: int = None,
#                  random: bool = False, use_truncation: bool = False, use_sample: bool = False,
#                  start_idx: int = 0, selective_chrs: list = None):
#         self.file_info = pd.read_csv(csv_file)
#         # self.file_info = self.file_info.sample(frac=1).reset_index(drop=True)
#         self.root_path = Path(root_path)
#         self.need_labels = need_labels
#         self.use_length_scale = use_length_scale
#         self.bias_power = bias_power
#         self.min_length = min_length
#         self.max_length = max_length
#         self.use_truncation = use_truncation
#         self.use_sample = use_sample
#         self.selective_chrs = selective_chrs
#         self.random = random
#         self.start_idx = start_idx
#         if K_mer not in K_mer_dict:
#             raise ValueError(f"Invalid K-mer type. Available options are: {list(K_mer_dict.keys())}")
#         self.N_mod = K_mer_dict[K_mer]["N_mod"]
#         if self.max_length == None:
#             self.max_length = K_mer_dict[K_mer]["seq_len"]
#         # Check and set whether 'celltype' and 'batch' are present
#         if need_labels and 'celltype' not in self.file_info.columns:
#             # Raise an error if 'need_labels' is True but 'celltype' column is missing
#             raise ValueError("The column 'celltype' is required in file_info but is missing.")
#         self.has_celltype = 'celltype' in self.file_info.columns and need_labels

#         if need_batch and 'batch' not in self.file_info.columns:
#             # Raise an error if 'need_batch' is True but 'batch' column is missing
#             raise ValueError("The column 'batch' is required in file_info but is missing.")
#         self.has_batch = 'batch' in self.file_info.columns and need_batch

#         self.need_analysis = need_analysis
#         if self.has_celltype:
#             with open(type_json_path, "r") as file:
#                 self.label_to_id = json.load(file)
#                 self.num_labels = len(set(self.label_to_id.values()))
#                 self.file_info = self.file_info[self.file_info['celltype'].isin(list(self.label_to_id.keys()))]

#         if self.has_batch:
#             with open(batch_type_json_path, "r") as file:
#                 self.batch_to_id = json.load(file)
#                 self.num_batches = len(self.batch_to_id)
#         else:
#             self.num_batches = None
                
#         self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
#         self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
#         self.end_token_id = tokenizer._convert_token_to_id("[SEP]")

#     def loadata(self, file_path):
#         try:
#             # Ensure file_path is a Path object
#             file_path = Path(file_path)
            
#             if file_path.suffix == '.npy':  # Check for .npy extension
#                 return np.load(file_path)
#             elif file_path.suffix == '.npz':  # Check for .npz extension
#                 return np.load(file_path)["data"]
#             else:
#                 print(f"Warning: File path {file_path} is not a .npy or .npz file. Returning an empty array.")
#                 return np.array([])
#         except Exception as e:
#             print(f"Error while loading file {file_path}: {e}. Returning an empty array.")
#             return np.array([])

#     def __len__(self) -> int:
#         """
#         Returns the number of samples in the dataset.
#         """
#         return len(self.file_info)

#     def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
#         print("###########################################################################################")
#         print("###########################################################################################")
#         """
#         Returns a single sample with either random or fixed sampling based on self.random_sample.

#         Args:
#             idx (int): Index of the sample.

#         Returns:
#             dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
#         """
#         token_path = self.root_path / self.file_info.iloc[idx]['file_name']
#         if not token_path.exists():
#             raise FileNotFoundError(f"Token file not found: {token_path}")

#         tokens_data = self.loadata(token_path)
#         unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
#         methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

#         ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
#         ratios_path = Path(ratios_path)
#         if not ratios_path.exists():
#             raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

#         methy_ratios = self.loadata(ratios_path)

#         # Create a mask to filter out NaN values in methy_ratios
#         valid_mask = ~np.isnan(methy_ratios)
#         print("###########################################################################################")
#         if self.need_analysis or self.selective_chrs is not None:
#             if self.need_analysis:
#                 positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
#                 positions_path = Path(positions_path)
#                 if not positions_path.exists():
#                     raise FileNotFoundError(f"Positions file not found: {positions_path}")
#                 positions = self.loadata(positions_path)
            
#             chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
#             print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             chrs_path = Path(chrs_path)
#             if not chrs_path.exists():
#                 raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
#             chrs = self.loadata(chrs_path)
            
#             if self.selective_chrs is not None:
#                 filter_chrs_idx = np.isin(chrs, self.selective_chrs)

#                 valid_mask_chrs = valid_mask & filter_chrs_idx

#                 unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
#                 methy_input_ids = methy_input_ids[valid_mask_chrs]
#                 methy_ratios = methy_ratios[valid_mask_chrs]
#                 if self.need_analysis:
#                     positions = positions[valid_mask_chrs]
#                 chrs = chrs[valid_mask_chrs]
#         else:
#             # Filter all the arrays using the valid_mask
#             unmethy_input_ids = unmethy_input_ids[valid_mask]
#             methy_input_ids = methy_input_ids[valid_mask]
#             methy_ratios = methy_ratios[valid_mask]

#         length = len(unmethy_input_ids)

#         if self.need_labels and self.use_length_scale:
#             r = random.random() ** self.bias_power
#             max_length = int(self.min_length + r * (self.max_length - self.min_length))
#             # max_length = random.randint(self.min_length, self.max_length)
#         else:
#             max_length = self.max_length

#         # if self.need_labels and self.use_length_scale:
#         #     if random.random() < 0.05:  # 5% chance
#         #         max_length = self.max_length
#         #     else:
#         #         r = random.random() ** self.bias_power
#         #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
#         # else:
#         #     max_length = self.max_length


#         if length >= max_length:
#             if self.use_sample:
#                 if self.random:
#                     # Random sampling
#                     sample_indices = np.random.choice(length, size=max_length, replace=False)
#                 else:
#                     # Fixed sampling: evenly spaced indices
#                     rng = np.random.default_rng(seed=42)
#                     sample_indices = rng.choice(length, size=max_length, replace=False)
#             elif self.use_truncation and self.random:
#                     # Random truncation: Select a contiguous range of max_length
#                     start_idx = np.random.randint(0, length - max_length + 1)
#                     sample_indices = np.arange(start_idx, start_idx + max_length)
#             else:
#                 sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
#             # Apply sampling
#             unmethy_input_ids = unmethy_input_ids[sample_indices]
#             methy_input_ids = methy_input_ids[sample_indices]
#             methy_ratios = methy_ratios[sample_indices]
            
#             if self.need_analysis or self.selective_chrs is not None:
#                 if self.need_analysis:
#                     positions = positions[sample_indices]
#                 chrs = chrs[sample_indices]

#         # Add start and end tokens
#         unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
#         methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
#         methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

#         sample = {
#             "unmethy_input_ids": unmethy_input_ids,
#             "methy_input_ids": methy_input_ids,
#             "methy_ratios": methy_ratios,
#         }
        
#         if self.need_analysis or self.selective_chrs is not None:
#             if self.need_analysis:
#                 positions = np.concatenate(([0], positions, [0]))
#                 sample["positions"] = positions
#             chrs = np.concatenate(([0], chrs, [0]))
#             sample["chrs"] = chrs
#             sample["file_names"] = self.file_info.iloc[idx]['file_name']

#         if self.has_celltype:
#             sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

#         if self.has_batch:
#             sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

#         return sample


class TokensRatiosDataset(Dataset):
    """
    A PyTorch Dataset for handling tokenized sequences and methylation ratios.

    Args:
        csv_file (str): Path to the CSV file containing file information.
        root_path (str): Root directory where data files are stored.
        K_mer (str): The K-mer type, must be a key in `K_mer_dict`.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 type_json_path: str = None, batch_type_json_path: str = None,
                 need_labels: bool = False, need_batch: bool = False,
                 use_length_scale: bool = False, bias_power: float=0.4, min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None, chrs_random: bool=False,
                 return_genome_prev_cpg: bool = False, token_path_genome_map: Dict[str, str] = None,
                 genome_cpg_resources: Dict[str, str] = None, genome_position_offset: int = 0):
        self.file_info = pd.read_csv(csv_file)
        # self.file_info = self.file_info.sample(frac=1).reset_index(drop=True)
        self.root_path = Path(root_path)
        self.need_labels = need_labels
        self.use_length_scale = use_length_scale
        self.bias_power = bias_power
        self.min_length = min_length
        self.max_length = max_length
        self.use_truncation = use_truncation
        self.use_sample = use_sample
        self.selective_chrs = selective_chrs
        self.chrs_random = chrs_random
        self.random = random
        self.start_idx = start_idx
        self.return_genome_prev_cpg = return_genome_prev_cpg
        self.token_path_genome_map = token_path_genome_map or {}
        self.genome_cpg_resources = {
            genome_key: Path(resource_path)
            for genome_key, resource_path in (genome_cpg_resources or {}).items()
        }
        self.genome_position_offset = genome_position_offset
        self._genome_cpg_cache = {}
        if K_mer not in K_mer_dict:
            raise ValueError(f"Invalid K-mer type. Available options are: {list(K_mer_dict.keys())}")
        self.N_mod = K_mer_dict[K_mer]["N_mod"]
        if self.max_length == None:
            self.max_length = K_mer_dict[K_mer]["seq_len"]
        # Check and set whether 'celltype' and 'batch' are present
        if need_labels and 'celltype' not in self.file_info.columns:
            # Raise an error if 'need_labels' is True but 'celltype' column is missing
            raise ValueError("The column 'celltype' is required in file_info but is missing.")
        self.has_celltype = 'celltype' in self.file_info.columns and need_labels

        if need_batch and 'batch' not in self.file_info.columns:
            # Raise an error if 'need_batch' is True but 'batch' column is missing
            raise ValueError("The column 'batch' is required in file_info but is missing.")
        self.has_batch = 'batch' in self.file_info.columns and need_batch

        if self.has_celltype:
            with open(type_json_path, "r") as file:
                self.label_to_id = json.load(file)
                self.num_labels = len(set(self.label_to_id.values()))
                self.file_info = self.file_info[self.file_info['celltype'].isin(list(self.label_to_id.keys()))]

        if self.has_batch:
            with open(batch_type_json_path, "r") as file:
                self.batch_to_id = json.load(file)
                self.num_batches = len(self.batch_to_id)
        else:
            self.num_batches = None
                
        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")

    def loadata(self, file_path):
        try:
            # Ensure file_path is a Path object
            file_path = Path(file_path)
            
            if file_path.suffix == '.npy':  # Check for .npy extension
                return np.load(file_path)
            elif file_path.suffix == '.npz':  # Check for .npz extension
                return np.load(file_path)["data"]
            else:
                print(f"Warning: File path {file_path} is not a .npy or .npz file. Returning an empty array.")
                return np.array([])
        except Exception as e:
            print(f"Error while loading file {file_path}: {e}. Returning an empty array.")
            return np.array([])

    def _normalize_cpg_reference_key(self, raw_key):
        key = str(raw_key).strip()
        key_lower = key.lower()
        for prefix in (
            "positions_",
            "position_",
            "cpg_positions_",
            "cpg_position_",
            "cpg_",
            "sites_",
            "site_",
        ):
            if key_lower.startswith(prefix):
                key = key[len(prefix):]
                break
        return normalize_chr_id(key)

    def _load_cpg_reference_from_array_file(self, resource_path: Path):
        if resource_path.suffix == ".npz":
            npz_data = np.load(resource_path, allow_pickle=True)
            try:
                if (
                    len(npz_data.files) == 1
                    and npz_data.files[0] == "data"
                    and getattr(npz_data["data"], "shape", None) == ()
                    and isinstance(npz_data["data"].item(), dict)
                ):
                    raw_mapping = npz_data["data"].item()
                else:
                    raw_mapping = {key: npz_data[key] for key in npz_data.files}
            finally:
                npz_data.close()
        elif resource_path.suffix == ".npy":
            raw_mapping = np.load(resource_path, allow_pickle=True)
            if getattr(raw_mapping, "shape", None) == () and hasattr(raw_mapping, "item"):
                raw_mapping = raw_mapping.item()
        else:
            raise ValueError(f"Unsupported CpG resource file: {resource_path}")

        if not isinstance(raw_mapping, dict):
            raise ValueError(
                f"CpG resource {resource_path} must be a dict-like npy/npz, got {type(raw_mapping)}"
            )

        cpg_reference = {}
        for raw_chr, raw_positions in raw_mapping.items():
            chr_key = self._normalize_cpg_reference_key(raw_chr)
            cpg_reference[chr_key] = np.unique(
                np.asarray(raw_positions, dtype=np.int64).reshape(-1)
            )
        return cpg_reference

    def _load_cpg_reference_from_fasta(self, resource_path: Path):
        cpg_reference = {}
        open_fn = gzip.open if resource_path.suffix == ".gz" else open

        current_chr = None
        current_positions = []
        current_offset = 0
        prev_char = ""

        with open_fn(resource_path, "rt") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith(">"):
                    if current_chr is not None:
                        cpg_reference[current_chr] = np.asarray(current_positions, dtype=np.int64)

                    current_chr = normalize_chr_id(line[1:].split()[0])
                    current_positions = []
                    current_offset = 0
                    prev_char = ""
                    continue

                if current_chr is None:
                    continue

                seq = line.upper()
                if prev_char == "C" and seq[0] == "G" and current_offset > 0:
                    current_positions.append(current_offset - 1)

                seq_array = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
                if seq_array.size > 1:
                    cpg_idx = np.flatnonzero(
                        (seq_array[:-1] == ord("C")) & (seq_array[1:] == ord("G"))
                    )
                    if cpg_idx.size:
                        current_positions.extend((current_offset + cpg_idx).tolist())

                prev_char = seq[-1]
                current_offset += len(seq)

        if current_chr is not None:
            cpg_reference[current_chr] = np.asarray(current_positions, dtype=np.int64)

        return cpg_reference

    def _resolve_genome_cpg_resource(self, token_path: Path):
        if len(self.genome_cpg_resources) == 1 and not self.token_path_genome_map:
            genome_key, resource_path = next(iter(self.genome_cpg_resources.items()))
            return genome_key, resource_path

        token_path_str = str(token_path)
        for keyword, mapped_value in sorted(
            self.token_path_genome_map.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if keyword in token_path_str:
                if mapped_value in self.genome_cpg_resources:
                    return mapped_value, self.genome_cpg_resources[mapped_value]
                return mapped_value, Path(mapped_value)

        raise KeyError(
            f"Cannot resolve genome CpG resource for token path: {token_path}. "
            "Please provide token_path_genome_map and genome_cpg_resources."
        )

    def _get_genome_cpg_reference(self, token_path: Path):
        genome_key, resource_path = self._resolve_genome_cpg_resource(token_path)
        if not resource_path.exists():
            raise FileNotFoundError(f"CpG resource not found for genome {genome_key}: {resource_path}")

        cache_key = str(resource_path.resolve())
        if cache_key not in self._genome_cpg_cache:
            suffixes = resource_path.suffixes
            is_fasta = (
                resource_path.suffix in {".fa", ".fasta", ".fna"}
                or suffixes[-2:] in ([".fa", ".gz"], [".fasta", ".gz"], [".fna", ".gz"])
            )
            if is_fasta:
                self._genome_cpg_cache[cache_key] = self._load_cpg_reference_from_fasta(resource_path)
            else:
                self._genome_cpg_cache[cache_key] = self._load_cpg_reference_from_array_file(resource_path)
        return self._genome_cpg_cache[cache_key]

    def _compute_prev_genome_prev_cpg_mask(self, token_path: Path, chrs, positions):
        prev_genome_prev_cpg = np.zeros(len(positions), dtype=bool)
        if len(positions) < 2:
            return prev_genome_prev_cpg

        cpg_reference = self._get_genome_cpg_reference(token_path)
        normalized_chrs = np.asarray([normalize_chr_id(chr_id) for chr_id in chrs], dtype=object)
        shifted_positions = np.asarray(positions, dtype=np.int64) + self.genome_position_offset

        for chr_id in np.unique(normalized_chrs):
            chr_mask = normalized_chrs == chr_id
            seq_indices = np.flatnonzero(chr_mask)
            if seq_indices.size < 2:
                continue

            reference_positions = cpg_reference.get(chr_id)
            if reference_positions is None or reference_positions.size == 0:
                continue

            seq_positions = shifted_positions[seq_indices]
            ref_ranks = np.searchsorted(reference_positions, seq_positions, side="left")

            in_reference = np.zeros(seq_positions.shape[0], dtype=bool)
            in_bounds = ref_ranks < reference_positions.size
            if np.any(in_bounds):
                in_reference[in_bounds] = (
                    reference_positions[ref_ranks[in_bounds]] == seq_positions[in_bounds]
                )

            prev_genome_prev_cpg[seq_indices[1:]] = (
                (seq_indices[1:] == (seq_indices[:-1] + 1))
                & in_reference[:-1]
                & in_reference[1:]
                & (ref_ranks[1:] == (ref_ranks[:-1] + 1))
            )

        return prev_genome_prev_cpg

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.file_info)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)

        positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
        positions_path = Path(positions_path)
        if not positions_path.exists():
            raise FileNotFoundError(f"Positions file not found: {positions_path}")
        positions = self.loadata(positions_path)
        
        chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
        chrs_path = Path(chrs_path)
        if not chrs_path.exists():
            raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
        chrs = self.loadata(chrs_path)
        
        if self.selective_chrs is not None:
            filter_chrs_idx = np.isin(chrs, self.selective_chrs)

            valid_mask_chrs = valid_mask & filter_chrs_idx

            unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
            methy_input_ids = methy_input_ids[valid_mask_chrs]
            methy_ratios = methy_ratios[valid_mask_chrs]

            positions = positions[valid_mask_chrs]
            chrs = chrs[valid_mask_chrs]

        genome_chrs = np.array(chrs, copy=True)

        # If self.chrs_random is True, shuffle the order of chromosomes
        if self.chrs_random:
            # Generate a random order for chromosomes
            unique_chrs = np.unique(chrs)  # Get the unique chromosome IDs
            shuffled_chrs = np.random.permutation(unique_chrs)  # Randomly shuffle the unique chromosome IDs
            
            # Create a mapping from original chromosomes to shuffled chromosomes
            chr_mapping = {chr_id: shuffled_chrs[i] for i, chr_id in enumerate(unique_chrs)}
            
            # Apply this mapping to the chrs array
            chrs = np.array([chr_mapping[chr_id] for chr_id in chrs])

            # Shuffle the other arrays accordingly
            # We can use the same indices to shuffle all arrays consistently
            shuffle_indices = np.argsort(chrs)  # Get the indices that would sort the chrs array

            unmethy_input_ids = unmethy_input_ids[shuffle_indices]
            methy_input_ids = methy_input_ids[shuffle_indices]
            methy_ratios = methy_ratios[shuffle_indices]
            positions = positions[shuffle_indices]
            chrs = chrs[shuffle_indices]  # Ensure chrs is also shuffled accordingly
            genome_chrs = genome_chrs[shuffle_indices]

        length = len(unmethy_input_ids)

        if self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length


        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            positions = positions[sample_indices]
            chrs = chrs[sample_indices]
            genome_chrs = genome_chrs[sample_indices]

        prev_is_genome_prev_cpg = None
        if self.return_genome_prev_cpg:
            prev_is_genome_prev_cpg = self._compute_prev_genome_prev_cpg_mask(
                token_path=token_path,
                chrs=genome_chrs,
                positions=positions,
            )

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        positions = np.concatenate(([0], positions, [0]))
        chrs = np.concatenate(([0], chrs, [0]))
        if prev_is_genome_prev_cpg is not None:
            prev_is_genome_prev_cpg = np.concatenate(([False], prev_is_genome_prev_cpg, [False]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
            "positions": positions,
            "chrs": chrs,
            "file_names": self.file_info.iloc[idx]['file_name'],
        }
        if prev_is_genome_prev_cpg is not None:
            sample["prev_is_genome_prev_cpg"] = prev_is_genome_prev_cpg

        if self.has_celltype:
            sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


class TokensRatiosArrayto01Dataset(TokensRatiosDataset):

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)

        positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
        positions_path = Path(positions_path)
        if not positions_path.exists():
            raise FileNotFoundError(f"Positions file not found: {positions_path}")
        positions = self.loadata(positions_path)
        
        chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
        chrs_path = Path(chrs_path)
        if not chrs_path.exists():
            raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
        chrs = self.loadata(chrs_path)
        
        if self.selective_chrs is not None:
            filter_chrs_idx = np.isin(chrs, self.selective_chrs)

            valid_mask_chrs = valid_mask & filter_chrs_idx

            unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
            methy_input_ids = methy_input_ids[valid_mask_chrs]
            methy_ratios = methy_ratios[valid_mask_chrs]

            positions = positions[valid_mask_chrs]
            chrs = chrs[valid_mask_chrs]

        genome_chrs = np.array(chrs, copy=True)

        # If self.chrs_random is True, shuffle the order of chromosomes
        if self.chrs_random:
            # Generate a random order for chromosomes
            unique_chrs = np.unique(chrs)  # Get the unique chromosome IDs
            shuffled_chrs = np.random.permutation(unique_chrs)  # Randomly shuffle the unique chromosome IDs
            
            # Create a mapping from original chromosomes to shuffled chromosomes
            chr_mapping = {chr_id: shuffled_chrs[i] for i, chr_id in enumerate(unique_chrs)}
            
            # Apply this mapping to the chrs array
            chrs = np.array([chr_mapping[chr_id] for chr_id in chrs])

            # Shuffle the other arrays accordingly
            # We can use the same indices to shuffle all arrays consistently
            shuffle_indices = np.argsort(chrs)  # Get the indices that would sort the chrs array

            unmethy_input_ids = unmethy_input_ids[shuffle_indices]
            methy_input_ids = methy_input_ids[shuffle_indices]
            methy_ratios = methy_ratios[shuffle_indices]
            positions = positions[shuffle_indices]
            chrs = chrs[shuffle_indices]  # Ensure chrs is also shuffled accordingly
            genome_chrs = genome_chrs[shuffle_indices]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length


        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            positions = positions[sample_indices]
            chrs = chrs[sample_indices]
            genome_chrs = genome_chrs[sample_indices]

        # methy_ratios = np.where(methy_ratios > 0.6, 1, 0)

        prev_is_genome_prev_cpg = None
        if self.return_genome_prev_cpg:
            prev_is_genome_prev_cpg = self._compute_prev_genome_prev_cpg_mask(
                token_path=token_path,
                chrs=genome_chrs,
                positions=positions,
            )

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        positions = np.concatenate(([0], positions, [0]))
        chrs = np.concatenate(([0], chrs, [0]))
        if prev_is_genome_prev_cpg is not None:
            prev_is_genome_prev_cpg = np.concatenate(([False], prev_is_genome_prev_cpg, [False]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
            "positions": positions,
            "chrs": chrs,
            "file_names": self.file_info.iloc[idx]['file_name'],
        }
        if prev_is_genome_prev_cpg is not None:
            sample["prev_is_genome_prev_cpg"] = prev_is_genome_prev_cpg

        if self.has_celltype:
            sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


class TokensRatiosPredictscRNAseqDataset(Dataset):
    """
    A PyTorch Dataset for handling tokenized sequences and methylation ratios.

    Args:
        csv_file (str): Path to the CSV file containing file information.
        root_path (str): Root directory where data files are stored.
        K_mer (str): The K-mer type, must be a key in `K_mer_dict`.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 gen_list_json_path: str = None, batch_type_json_path: str = None,
                 type_json_path: str = None, need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None):
        self.file_info = pd.read_csv(csv_file)
        # self.file_info = self.file_info.sample(frac=1).reset_index(drop=True)
        self.root_path = Path(root_path)
        self.need_labels = need_labels
        self.use_length_scale = use_length_scale
        self.bias_power = bias_power
        self.min_length = min_length
        self.max_length = max_length
        self.use_truncation = use_truncation
        self.use_sample = use_sample
        self.selective_chrs = selective_chrs
        self.random = random
        self.start_idx = start_idx
        if K_mer not in K_mer_dict:
            raise ValueError(f"Invalid K-mer type. Available options are: {list(K_mer_dict.keys())}")
        self.N_mod = K_mer_dict[K_mer]["N_mod"]
        if self.max_length == None:
            self.max_length = K_mer_dict[K_mer]["seq_len"]
        # Check and set whether 'normalized_expression' and 'batch' are present
        # if need_labels and 'normalized_expression' not in self.file_info.columns:
        #     # Raise an error if 'need_labels' is True but 'normalized_expression' column is missing
        #     raise ValueError("The column 'normalized_expression' is required in file_info but is missing.")
        # self.has_normalized_expression = 'normalized_expression' in self.file_info.columns
        self.has_celltype = 'celltype' in self.file_info.columns

        if self.has_celltype and type_json_path is not None:
            with open(type_json_path, "r") as file:
                self.label_to_id = json.load(file)
                self.file_info = self.file_info[self.file_info['celltype'].isin(list(self.label_to_id.keys()))]

        if need_batch and 'batch' not in self.file_info.columns:
            # Raise an error if 'need_batch' is True but 'batch' column is missing
            raise ValueError("The column 'batch' is required in file_info but is missing.")
        self.has_batch = 'batch' in self.file_info.columns and need_batch

        self.need_analysis = need_analysis
        if self.need_labels:
            with open(gen_list_json_path, "r") as file:
                self.gen_list = json.load(file)
                self.num_labels = len(self.gen_list)
                # self.file_info = self.file_info[self.file_info['normalized_expression'].isin(list(self.label_to_id.keys()))]

        if self.has_batch:
            with open(batch_type_json_path, "r") as file:
                self.batch_to_id = json.load(file)
                self.num_batches = len(self.batch_to_id)
        else:
            self.num_batches = None
                
        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")

    def loadata(self, file_path):
        try:
            # Ensure file_path is a Path object
            file_path = Path(file_path)
            
            if file_path.suffix == '.npy':  # Check for .npy extension
                return np.load(file_path)
            elif file_path.suffix == '.npz':  # Check for .npz extension
                return np.load(file_path)["data"]
            else:
                print(f"Warning: File path {file_path} is not a .npy or .npz file. Returning an empty array.")
                return np.array([])
        except Exception as e:
            print(f"Error while loading file {file_path}: {e}. Returning an empty array.")
            return np.array([])

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.file_info)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)
        
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                positions_path = Path(positions_path)
                if not positions_path.exists():
                    raise FileNotFoundError(f"Positions file not found: {positions_path}")
                positions = self.loadata(positions_path)
            
            chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
            chrs_path = Path(chrs_path)
            if not chrs_path.exists():
                raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
            chrs = self.loadata(chrs_path)
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)

                valid_mask_chrs = valid_mask & filter_chrs_idx

                if self.need_analysis:
                    # region_start = 65_263_738 - 5_000_000 #128_748_840 #- 1000
                    # region_end   = 65_276_556 + 5_000_000 #128_753_204 #+ 1000
                    # filter_poses_idx = (positions >= region_start) & (positions <= region_end)
                    # valid_mask_chrs = filter_poses_idx & valid_mask_chrs
                    positions = positions[valid_mask_chrs]

                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length


        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                if self.need_analysis:
                    positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.file_info.iloc[idx]['file_name']

        if self.need_labels:
            sample["labels"] = self.file_info.iloc[idx][self.gen_list].values.astype(np.float32)

        if self.has_celltype:
            sample["celltype"] = self.label_to_id[self.file_info.iloc[idx]['celltype']] #
            sample["celltype_name"] = self.file_info.iloc[idx]['celltype']

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


def sample_nonzero_indices(labels: np.ndarray, k: int) -> np.ndarray:
    """
    从一个样本的 label 向量中采样 k 个非零元素的索引（不重复采样，数量不足则重复补齐）。

    参数:
        labels: np.ndarray of shape (G,)，二值向量（0/1），表示哪些基因被表达
        k: int，采样数量

    返回:
        sampled_indices: np.ndarray of shape (k,)，表示在原始 labels 中的索引位置
    """
    # 找出所有非零的索引位置
    nonzero_indices = np.flatnonzero(labels)

    if len(nonzero_indices) == 0:
        # 没有非零元素，返回全是 0 的索引（可自定义处理）
        return np.zeros(k, dtype=int)
    elif len(nonzero_indices) >= k:
        # 非零位置充足，随机选择 k 个不重复
        sampled_indices = np.random.choice(nonzero_indices, size=k, replace=False)
    else:
        # 不足 k 个，允许重复补齐
        sampled_indices = np.random.choice(nonzero_indices, size=k, replace=True)

    return sampled_indices


class TokensRatiosPredictscALLRNAseqDataset(Dataset):
    """
    A PyTorch Dataset for handling tokenized sequences and methylation ratios.

    Args:
        csv_file (str): Path to the CSV file containing file information.
        root_path (str): Root directory where data files are stored.
        K_mer (str): The K-mer type, must be a key in `K_mer_dict`.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 gen_list_json_path: str = None, batch_type_json_path: str = None,
                 type_json_path: str = None, need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, min_length: int = 50000, 
                 max_length: int = None, k: int=8, random: bool = False, use_truncation: bool = False, 
                 use_sample: bool = False, start_idx: int = 0, selective_chrs: list = None):
        self.file_info = pd.read_csv(csv_file)
        self.k = k
        # self.file_info = self.file_info.sample(frac=1).reset_index(drop=True)
        self.root_path = Path(root_path)
        self.need_labels = need_labels
        self.use_length_scale = use_length_scale
        self.bias_power = bias_power
        self.min_length = min_length
        self.max_length = max_length
        self.use_truncation = use_truncation
        self.use_sample = use_sample
        self.selective_chrs = selective_chrs
        self.random = random
        self.start_idx = start_idx
        if K_mer not in K_mer_dict:
            raise ValueError(f"Invalid K-mer type. Available options are: {list(K_mer_dict.keys())}")
        self.N_mod = K_mer_dict[K_mer]["N_mod"]
        if self.max_length == None:
            self.max_length = K_mer_dict[K_mer]["seq_len"]
        # Check and set whether 'normalized_expression' and 'batch' are present
        # if need_labels and 'normalized_expression' not in self.file_info.columns:
        #     # Raise an error if 'need_labels' is True but 'normalized_expression' column is missing
        #     raise ValueError("The column 'normalized_expression' is required in file_info but is missing.")
        # self.has_normalized_expression = 'normalized_expression' in self.file_info.columns
        self.has_celltype = 'celltype' in self.file_info.columns

        if self.has_celltype and type_json_path is not None:
            with open(type_json_path, "r") as file:
                self.label_to_id = json.load(file)
                self.file_info = self.file_info[self.file_info['celltype'].isin(list(self.label_to_id.keys()))]

        if need_batch and 'batch' not in self.file_info.columns:
            # Raise an error if 'need_batch' is True but 'batch' column is missing
            raise ValueError("The column 'batch' is required in file_info but is missing.")
        self.has_batch = 'batch' in self.file_info.columns and need_batch

        self.need_analysis = need_analysis
        if self.need_labels:
            with open(gen_list_json_path, "r") as file:
                self.gen_list = json.load(file)
                self.num_labels = len(self.gen_list)
                # self.file_info = self.file_info[self.file_info['normalized_expression'].isin(list(self.label_to_id.keys()))]

        if self.has_batch:
            with open(batch_type_json_path, "r") as file:
                self.batch_to_id = json.load(file)
                self.num_batches = len(self.batch_to_id)
        else:
            self.num_batches = None
                
        self.pad_token_id = tokenizer._convert_token_to_id("[PAD]")
        self.start_token_id = tokenizer._convert_token_to_id("[BOS]")
        self.end_token_id = tokenizer._convert_token_to_id("[SEP]")

    def loadata(self, file_path):
        try:
            # Ensure file_path is a Path object
            file_path = Path(file_path)
            
            if file_path.suffix == '.npy':  # Check for .npy extension
                return np.load(file_path)
            elif file_path.suffix == '.npz':  # Check for .npz extension
                return np.load(file_path)["data"]
            else:
                print(f"Warning: File path {file_path} is not a .npy or .npz file. Returning an empty array.")
                return np.array([])
        except Exception as e:
            print(f"Error while loading file {file_path}: {e}. Returning an empty array.")
            return np.array([])

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.file_info)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)
        
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                positions_path = Path(positions_path)
                if not positions_path.exists():
                    raise FileNotFoundError(f"Positions file not found: {positions_path}")
                positions = self.loadata(positions_path)
            
            chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
            chrs_path = Path(chrs_path)
            if not chrs_path.exists():
                raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
            chrs = self.loadata(chrs_path)
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)

                valid_mask_chrs = valid_mask & filter_chrs_idx

                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                if self.need_analysis:
                    positions = positions[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length


        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                if self.need_analysis:
                    positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.file_info.iloc[idx]['file_name']

        if self.need_labels:
            all_gene_labels = self.file_info.iloc[idx][self.gen_list].values.astype(np.float32)
            sampled_gene_indices = sample_nonzero_indices(all_gene_labels, self.k)
            # print("######################", sampled_gene_indices)
            sampled_labels = all_gene_labels[sampled_gene_indices]
            sample["labels"] = sampled_labels
            sample["sampled_gene_indices"] = sampled_gene_indices

        if self.has_celltype:
            sample["celltype"] = self.label_to_id[self.file_info.iloc[idx]['celltype']] #
            sample["celltype_name"] = self.file_info.iloc[idx]['celltype']

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


class TokensRatiosLoadALLDataset(TokensRatiosDataset):
    """
    A subclass of TokensRatiosDataset that loads all data at once during initialization.
    This class overrides the necessary methods to load data in bulk and access it directly.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 type_json_path: str = None, batch_type_json_path: str = None,
                 need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, 
                 min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None):
        # Initialize the parent class
        super().__init__(csv_file, root_path, K_mer, tokenizer, 
                         type_json_path, batch_type_json_path,
                         need_labels, need_batch,
                         need_analysis, use_length_scale, 
                         bias_power, min_length, max_length,
                         random, use_truncation, use_sample,
                         start_idx, selective_chrs)

        # Load all data at once in the initialization
        self.tokens_data = []
        self.methy_ratios_data = []
        self.chrs_data = []
        self.positions_data = [] if need_analysis else None
        self.files = self.file_info['file_name'].tolist()

        for file_name in self.files:
            token_path = self.root_path / file_name
            if token_path.exists():
                tokens = self.loadata(token_path)
                self.tokens_data.append(tokens)

                ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
                ratios_path = Path(ratios_path)
                if ratios_path.exists():
                    ratios = self.loadata(ratios_path)
                    self.methy_ratios_data.append(ratios)

                if need_analysis:
                    positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                    positions_path = Path(positions_path)
                    if positions_path.exists():
                        positions = self.loadata(positions_path)
                        self.positions_data.append(positions)

                chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
                chrs_path = Path(chrs_path)
                if chrs_path.exists():
                    chrs = self.loadata(chrs_path)
                    self.chrs_data.append(chrs)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Override to directly access the pre-loaded data.
        """
        # Access the pre-loaded data
        tokens_data = self.tokens_data[idx]
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)

        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        methy_ratios = self.methy_ratios_data[idx]

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)

        # Apply the selective chromosome filtering if needed
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions = self.positions_data[idx] if self.positions_data else np.array([])

            chrs = self.chrs_data[idx]
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)
                
                # Further filter the data based on chromosome selection
                valid_mask_chrs = valid_mask & filter_chrs_idx
                
                # Apply the combined mask for both NaN and selective chromosomes
                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                if self.need_analysis:
                    positions = positions[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length

        if length >= max_length:
            if self.use_sample:
                if self.random:
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.files[idx]

        if self.has_celltype:
            sample["labels"] = self.label_to_id[self.file_info.iloc[idx]['celltype']]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


class TokensRatiosLoadALLscRNAseqDataset(TokensRatiosDataset):
    """
    A subclass of TokensRatiosDataset that loads all data at once during initialization.
    This class overrides the necessary methods to load data in bulk and access it directly.
    """
    def __init__(self, csv_file: str, root_path: str, K_mer: str, tokenizer, 
                 type_json_path: str = None, batch_type_json_path: str = None,
                 need_labels: bool = False, need_batch: bool = False,
                 need_analysis=False, use_length_scale: bool = False, bias_power: float=0.4, 
                 min_length: int = 50000, max_length: int = None,
                 random: bool = False, use_truncation: bool = False, use_sample: bool = False,
                 start_idx: int = 0, selective_chrs: list = None):
        # Initialize the parent class
        super().__init__(csv_file, root_path, K_mer, tokenizer, 
                         type_json_path, batch_type_json_path,
                         need_labels, need_batch,
                         need_analysis, use_length_scale, 
                         bias_power, min_length, max_length,
                         random, use_truncation, use_sample,
                         start_idx, selective_chrs)

        # Load all data at once in the initialization
        self.tokens_data = []
        self.methy_ratios_data = []
        self.labels = []
        self.chrs_data = []
        self.positions_data = []

        for _, row in self.file_info.iterrows():
            token_path = self.root_path / row["file_name"]
            # print(token_path)
            if not token_path.exists():
                # print("continue")
                continue

            tokens = self.loadata(token_path)

            # 加载配套数据
            ratios_path = Path(str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz"))
            chrs_path = Path(str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz"))
            pos_path  = Path(str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz"))

            if not (ratios_path.exists() and chrs_path.exists() and pos_path.exists()):
                continue

            ratios = self.loadata(ratios_path)
            chrs = self.loadata(chrs_path)
            positions = self.loadata(pos_path)

            # Step 1: 选中染色体
            target_chr = row["chr"]
            chr_mask = (chrs == target_chr)

            tokens = tokens[chr_mask]
            ratios = ratios[chr_mask]
            positions = positions[chr_mask]

            center = (row["start"] + row["end"]) // 2
            half_len = 8000000000 // 2
            start_pos = center - half_len
            end_pos = center + half_len

            center_mask = (positions >= start_pos) & (positions < end_pos)

            tokens = tokens[center_mask]
            ratios = ratios[center_mask]
            positions = positions[center_mask]

            # 丢弃空段
            if len(tokens) == 0:
                continue

            # 保存
            self.tokens_data.append(tokens)
            self.methy_ratios_data.append(ratios)
            self.positions_data.append(positions)
            self.chrs_data.append([target_chr] * len(tokens))
            self.labels.append(row["normalized_expression"])

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Override to directly access the pre-loaded data.
        """
        # Access the pre-loaded data
        # print(len(self.tokens_data[idx]), idx, "!!!!!!!!!!!!!!!!!!!!!")
        tokens_data = self.tokens_data[idx]
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)

        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        methy_ratios = self.methy_ratios_data[idx]

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)

        # Apply the selective chromosome filtering if needed
        if self.need_analysis:
            if self.need_analysis:
                positions = self.positions_data[idx] if self.positions_data else np.array([])
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.need_labels and self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        # if self.need_labels and self.use_length_scale:
        #     if random.random() < 0.05:  # 5% chance
        #         max_length = self.max_length
        #     else:
        #         r = random.random() ** self.bias_power
        #         max_length = int(self.min_length + r * (self.max_length - self.min_length))
        # else:
        #     max_length = self.max_length

        if length >= max_length:
            if self.use_sample:
                if self.random:
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                positions = positions[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            sample["file_names"] = self.files[idx]

        # if self.has_celltype:
        sample["labels"] = self.labels[idx]

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample

@dataclass
class scdnam_collate_TokensRatios:
    """
    A PyTorch collate function for batching tokenized sequences and methylation ratios.

    Args:
        tokenizer: A tokenizer object with methods `_convert_token_to_id`.
        K_mer (str): The K-mer type, must match the dataset configuration.
    """
    tokenizer: object

    def __post_init__(self):
        self.pad_token_id = self.tokenizer._convert_token_to_id("[PAD]")

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed_unmethy_inputs, processed_methy_inputs, processed_methy_ratios, labels, batch_labels = [], [], [], [], []
        processed_positions, processed_chrs, processed_prev_is_genome_prev_cpg, file_names = [], [], [], []

        for item in instances:
            unmethy_input_ids = item['unmethy_input_ids']
            methy_input_ids = item['methy_input_ids']
            methy_ratios = item['methy_ratios']

            processed_unmethy_inputs.append(torch.tensor(unmethy_input_ids, dtype=torch.long))
            processed_methy_inputs.append(torch.tensor(methy_input_ids, dtype=torch.long))
            processed_methy_ratios.append(torch.tensor(methy_ratios, dtype=torch.float))

            if 'labels' in item:
                labels.append(item['labels'])

            if 'batch_labels' in item:
                batch_labels.append(item['batch_labels'])
                
            if 'positions' in item:
                processed_positions.append(torch.tensor(item['positions'], dtype=torch.long))

            if 'chrs' in item:
                processed_chrs.append(torch.tensor(item['chrs'], dtype=torch.long))

            if 'prev_is_genome_prev_cpg' in item:
                processed_prev_is_genome_prev_cpg.append(
                    torch.tensor(item['prev_is_genome_prev_cpg'], dtype=torch.bool)
                )
                
            if 'file_names' in item:
                file_names.append(item['file_names'])

        unmethy_inputs_padded = pad_sequence(processed_unmethy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_inputs_padded = pad_sequence(processed_methy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_ratios_padded = pad_sequence(processed_methy_ratios, batch_first=True, padding_value=-1.0)
        positions_padded = pad_sequence(processed_positions, batch_first=True, padding_value=-1) if processed_positions else None
        chrs_padded = pad_sequence(processed_chrs, batch_first=True, padding_value=-1) if processed_chrs else None
        prev_is_genome_prev_cpg_padded = (
            pad_sequence(processed_prev_is_genome_prev_cpg, batch_first=True, padding_value=False)
            if processed_prev_is_genome_prev_cpg else None
        )
        file_names = file_names if file_names else None

        labels_tensor = torch.tensor(labels, dtype=torch.long) if labels else None
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long) if batch_labels else None

        return {
            'unmethy_input_ids': unmethy_inputs_padded,
            'methy_input_ids': methy_inputs_padded,
            'methy_ratios': methy_ratios_padded,
            'labels': labels_tensor,
            'batch_labels': batch_labels_tensor,
            "positions": positions_padded,
            "chrs": chrs_padded,
            "prev_is_genome_prev_cpg": prev_is_genome_prev_cpg_padded,
            "file_names": file_names,
        }


scDNAm_collate_TokensRatios = scdnam_collate_TokensRatios


@dataclass
class scdnam_collate_PredictscALLRNAseqRatios:
    """
    A PyTorch collate function for batching tokenized sequences and methylation ratios.

    Args:
        tokenizer: A tokenizer object with methods `_convert_token_to_id`.
        K_mer (str): The K-mer type, must match the dataset configuration.
    """
    tokenizer: object

    def __post_init__(self):
        self.pad_token_id = self.tokenizer._convert_token_to_id("[PAD]")

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed_unmethy_inputs, processed_methy_inputs, processed_methy_ratios, labels, batch_labels = [], [], [], [], []
        processed_positions, processed_chrs, processed_prev_is_genome_prev_cpg, file_names, celltype, celltype_name, sampled_gene_indices = [], [], [], [], [], [], []

        for item in instances:
            unmethy_input_ids = item['unmethy_input_ids']
            methy_input_ids = item['methy_input_ids']
            methy_ratios = item['methy_ratios']

            processed_unmethy_inputs.append(torch.tensor(unmethy_input_ids, dtype=torch.long))
            processed_methy_inputs.append(torch.tensor(methy_input_ids, dtype=torch.long))
            processed_methy_ratios.append(torch.tensor(methy_ratios, dtype=torch.float))

            if 'sampled_gene_indices' in item:
                sampled_gene_indices.append(item['sampled_gene_indices'])

            if 'labels' in item:
                labels.append(item['labels'])

            if 'batch_labels' in item:
                batch_labels.append(item['batch_labels'])
                
            if 'positions' in item:
                processed_positions.append(torch.tensor(item['positions'], dtype=torch.long))

            if 'chrs' in item:
                processed_chrs.append(torch.tensor(item['chrs'], dtype=torch.long))

            if 'prev_is_genome_prev_cpg' in item:
                processed_prev_is_genome_prev_cpg.append(
                    torch.tensor(item['prev_is_genome_prev_cpg'], dtype=torch.bool)
                )
                
            if 'file_names' in item:
                file_names.append(item['file_names'])

            if 'celltype' in item:
                celltype.append(item['celltype'])

            if 'celltype_name' in item:
                celltype_name.append(item['celltype_name'])

        unmethy_inputs_padded = pad_sequence(processed_unmethy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_inputs_padded = pad_sequence(processed_methy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_ratios_padded = pad_sequence(processed_methy_ratios, batch_first=True, padding_value=-1.0)
        positions_padded = pad_sequence(processed_positions, batch_first=True, padding_value=-1) if processed_positions else None
        chrs_padded = pad_sequence(processed_chrs, batch_first=True, padding_value=-1) if processed_chrs else None
        prev_is_genome_prev_cpg_padded = (
            pad_sequence(processed_prev_is_genome_prev_cpg, batch_first=True, padding_value=False)
            if processed_prev_is_genome_prev_cpg else None
        )
        file_names = file_names if file_names else None
        celltype = torch.tensor(celltype, dtype=torch.long) if celltype else None
        
        if sampled_gene_indices:
            sampled_gene_indices = torch.tensor(np.array(sampled_gene_indices), dtype=torch.long)
        else:
            sampled_gene_indices = None

        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float) if labels else None
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long) if batch_labels else None

        return {
            'unmethy_input_ids': unmethy_inputs_padded,
            'methy_input_ids': methy_inputs_padded,
            'methy_ratios': methy_ratios_padded,
            'labels': labels_tensor,
            'batch_labels': batch_labels_tensor,
            "positions": positions_padded,
            "chrs": chrs_padded,
            "prev_is_genome_prev_cpg": prev_is_genome_prev_cpg_padded,
            "file_names": file_names,
            "celltype": celltype,
            "celltype_name": celltype_name,
            "sampled_gene_indices": sampled_gene_indices
        }


@dataclass
class scdnam_collate_PredictscRNAseqRatios:
    """
    A PyTorch collate function for batching tokenized sequences and methylation ratios.

    Args:
        tokenizer: A tokenizer object with methods `_convert_token_to_id`.
        K_mer (str): The K-mer type, must match the dataset configuration.
    """
    tokenizer: object

    def __post_init__(self):
        self.pad_token_id = self.tokenizer._convert_token_to_id("[PAD]")

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed_unmethy_inputs, processed_methy_inputs, processed_methy_ratios, labels, batch_labels = [], [], [], [], []
        processed_positions, processed_chrs, processed_prev_is_genome_prev_cpg, file_names, celltype, celltype_name = [], [], [], [], [], []

        for item in instances:
            unmethy_input_ids = item['unmethy_input_ids']
            methy_input_ids = item['methy_input_ids']
            methy_ratios = item['methy_ratios']

            processed_unmethy_inputs.append(torch.tensor(unmethy_input_ids, dtype=torch.long))
            processed_methy_inputs.append(torch.tensor(methy_input_ids, dtype=torch.long))
            processed_methy_ratios.append(torch.tensor(methy_ratios, dtype=torch.float))

            if 'labels' in item:
                labels.append(item['labels'])

            if 'batch_labels' in item:
                batch_labels.append(item['batch_labels'])
                
            if 'positions' in item:
                processed_positions.append(torch.tensor(item['positions'], dtype=torch.long))

            if 'chrs' in item:
                processed_chrs.append(torch.tensor(item['chrs'], dtype=torch.long))

            if 'prev_is_genome_prev_cpg' in item:
                processed_prev_is_genome_prev_cpg.append(
                    torch.tensor(item['prev_is_genome_prev_cpg'], dtype=torch.bool)
                )
                
            if 'file_names' in item:
                file_names.append(item['file_names'])

            if 'celltype' in item:
                celltype.append(item['celltype'])

            if 'celltype_name' in item:
                celltype_name.append(item['celltype_name'])

        unmethy_inputs_padded = pad_sequence(processed_unmethy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_inputs_padded = pad_sequence(processed_methy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_ratios_padded = pad_sequence(processed_methy_ratios, batch_first=True, padding_value=-1.0)
        positions_padded = pad_sequence(processed_positions, batch_first=True, padding_value=-1) if processed_positions else None
        chrs_padded = pad_sequence(processed_chrs, batch_first=True, padding_value=-1) if processed_chrs else None
        prev_is_genome_prev_cpg_padded = (
            pad_sequence(processed_prev_is_genome_prev_cpg, batch_first=True, padding_value=False)
            if processed_prev_is_genome_prev_cpg else None
        )
        file_names = file_names if file_names else None
        celltype = torch.tensor(celltype, dtype=torch.long) if celltype else None

        labels_tensor = torch.tensor(np.array(labels), dtype=torch.float) if labels else None
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long) if batch_labels else None

        return {
            'unmethy_input_ids': unmethy_inputs_padded,
            'methy_input_ids': methy_inputs_padded,
            'methy_ratios': methy_ratios_padded,
            'labels': labels_tensor,
            'batch_labels': batch_labels_tensor,
            "positions": positions_padded,
            "chrs": chrs_padded,
            "prev_is_genome_prev_cpg": prev_is_genome_prev_cpg_padded,
            "file_names": file_names,
            "celltype": celltype,
            "celltype_name": celltype_name
        }


class TokensRatiosDatasetDeconv(TokensRatiosDataset):

    # Define a function to assign labels based on the file name
    def assign_label(self, file_name):
        # Extract the number from the second part of the file name
        file_number = int(file_name.split("_")[1])
        
        # Assign label based on the range of the file number
        if 0 <= file_number <= 99:
            return 0.0
        elif 100 <= file_number <= 199:
            return 0.1
        elif 200 <= file_number <= 299:
            return 0.2
        elif 300 <= file_number <= 399:
            return 0.3
        elif 400 <= file_number <= 499:
            return 0.4
        elif 500 <= file_number <= 599:
            return 0.5
        elif 600 <= file_number <= 699:
            return 0.6
        elif 700 <= file_number <= 799:
            return 0.7
        elif 800 <= file_number <= 899:
            return 0.8
        else:
            return None  # You can adjust this if you have other ranges

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns a single sample with either random or fixed sampling based on self.random_sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with keys `unmethy_input_ids`, `methy_ratios`, and optionally `labels`.
        """
        token_path = self.root_path / self.file_info.iloc[idx]['file_name']
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")

        tokens_data = self.loadata(token_path)
        unmethy_input_ids = np.where(tokens_data > 6, (tokens_data - 7) % self.N_mod + 7, tokens_data)
        
        methy_input_ids = np.where(unmethy_input_ids > 6, unmethy_input_ids + 1, unmethy_input_ids)

        ratios_path = str(token_path).replace("tokens", "ratios").replace(".npy", "_ratios.npy").replace(".npz", "_ratios.npz")
        ratios_path = Path(ratios_path)
        if not ratios_path.exists():
            raise FileNotFoundError(f"Ratios file not found: {ratios_path}")

        methy_ratios = self.loadata(ratios_path)

        # Create a mask to filter out NaN values in methy_ratios
        valid_mask = ~np.isnan(methy_ratios)
        
        if self.need_analysis or self.selective_chrs is not None:
            if self.need_analysis:
                positions_path = str(token_path).replace("tokens", "positions").replace(".npy", "_positions.npy").replace(".npz", "_positions.npz")
                positions_path = Path(positions_path)
                if not positions_path.exists():
                    raise FileNotFoundError(f"Positions file not found: {positions_path}")
                positions = self.loadata(positions_path)
            
            chrs_path = str(token_path).replace("tokens", "chrs").replace(".npy", "_chrs.npy").replace(".npz", "_chrs.npz")
            chrs_path = Path(chrs_path)
            if not chrs_path.exists():
                raise FileNotFoundError(f"Chrs file not found: {chrs_path}")
            chrs = self.loadata(chrs_path)
            
            if self.selective_chrs is not None:
                filter_chrs_idx = np.isin(chrs, self.selective_chrs)

                valid_mask_chrs = valid_mask & filter_chrs_idx

                unmethy_input_ids = unmethy_input_ids[valid_mask_chrs]
                methy_input_ids = methy_input_ids[valid_mask_chrs]
                methy_ratios = methy_ratios[valid_mask_chrs]
                if self.need_analysis:
                    positions = positions[valid_mask_chrs]
                chrs = chrs[valid_mask_chrs]
        else:
            # Filter all the arrays using the valid_mask
            unmethy_input_ids = unmethy_input_ids[valid_mask]
            methy_input_ids = methy_input_ids[valid_mask]
            methy_ratios = methy_ratios[valid_mask]

        length = len(unmethy_input_ids)

        if self.use_length_scale:
            r = random.random() ** self.bias_power
            max_length = int(self.min_length + r * (self.max_length - self.min_length))
            # max_length = random.randint(self.min_length, self.max_length)
        else:
            max_length = self.max_length

        if length >= max_length:
            if self.use_sample:
                if self.random:
                    # Random sampling
                    sample_indices = np.random.choice(length, size=max_length, replace=False)
                else:
                    # Fixed sampling: evenly spaced indices
                    rng = np.random.default_rng(seed=42)
                    sample_indices = rng.choice(length, size=max_length, replace=False)
            elif self.use_truncation and self.random:
                    # Random truncation: Select a contiguous range of max_length
                    start_idx = np.random.randint(0, length - max_length + 1)
                    sample_indices = np.arange(start_idx, start_idx + max_length)
            else:
                sample_indices = np.arange(self.start_idx, self.start_idx + max_length)
                
            # Apply sampling
            unmethy_input_ids = unmethy_input_ids[sample_indices]
            methy_input_ids = methy_input_ids[sample_indices]
            methy_ratios = methy_ratios[sample_indices]
            
            if self.need_analysis:
                if self.need_analysis:
                    positions = positions[sample_indices]
                chrs = chrs[sample_indices]

        # Add start and end tokens
        unmethy_input_ids = np.concatenate(([self.start_token_id], unmethy_input_ids, [self.end_token_id]))
        methy_input_ids = np.concatenate(([self.start_token_id], methy_input_ids, [self.end_token_id]))
        methy_ratios = np.concatenate(([1.0], methy_ratios, [1.0]))

        sample = {
            "unmethy_input_ids": unmethy_input_ids,
            "methy_input_ids": methy_input_ids,
            "methy_ratios": methy_ratios,
        }
        
        if self.need_analysis:
            if self.need_analysis:
                positions = np.concatenate(([0], positions, [0]))
                sample["positions"] = positions
            chrs = np.concatenate(([0], chrs, [0]))
            sample["chrs"] = chrs
            sample["file_names"] = self.file_info.iloc[idx]['file_name']

        # if self.has_celltype:
        sample["labels"] = self.assign_label(self.file_info.iloc[idx]['file_name'])

        if self.has_batch:
            sample["batch_labels"] = self.batch_to_id[self.file_info.iloc[idx]['batch']]

        return sample


class scdnam_collate_DeconvTokensRatios(scdnam_collate_TokensRatios):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        processed_unmethy_inputs, processed_methy_inputs, processed_methy_ratios, labels, batch_labels = [], [], [], [], []
        processed_positions, processed_chrs, processed_prev_is_genome_prev_cpg, file_names = [], [], [], []

        for item in instances:
            unmethy_input_ids = item['unmethy_input_ids']
            methy_input_ids = item['methy_input_ids']
            methy_ratios = item['methy_ratios']

            processed_unmethy_inputs.append(torch.tensor(unmethy_input_ids, dtype=torch.long))
            processed_methy_inputs.append(torch.tensor(methy_input_ids, dtype=torch.long))
            processed_methy_ratios.append(torch.tensor(methy_ratios, dtype=torch.float))

            if 'labels' in item:
                labels.append(item['labels'])

            if 'batch_labels' in item:
                batch_labels.append(item['batch_labels'])
                
            if 'positions' in item:
                processed_positions.append(torch.tensor(item['positions'], dtype=torch.long))

            if 'chrs' in item:
                processed_chrs.append(torch.tensor(item['chrs'], dtype=torch.long))

            if 'prev_is_genome_prev_cpg' in item:
                processed_prev_is_genome_prev_cpg.append(
                    torch.tensor(item['prev_is_genome_prev_cpg'], dtype=torch.bool)
                )
                
            if 'file_names' in item:
                file_names.append(item['file_names'])

        unmethy_inputs_padded = pad_sequence(processed_unmethy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_inputs_padded = pad_sequence(processed_methy_inputs, batch_first=True, padding_value=self.pad_token_id)
        methy_ratios_padded = pad_sequence(processed_methy_ratios, batch_first=True, padding_value=-1.0)
        positions_padded = pad_sequence(processed_positions, batch_first=True, padding_value=-1) if processed_positions else None
        chrs_padded = pad_sequence(processed_chrs, batch_first=True, padding_value=-1) if processed_chrs else None
        prev_is_genome_prev_cpg_padded = (
            pad_sequence(processed_prev_is_genome_prev_cpg, batch_first=True, padding_value=False)
            if processed_prev_is_genome_prev_cpg else None
        )
        file_names = file_names if file_names else None

        labels_tensor = torch.tensor(labels, dtype=torch.float) if labels else None
        batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.float) if batch_labels else None

        return {
            'unmethy_input_ids': unmethy_inputs_padded,
            'methy_input_ids': methy_inputs_padded,
            'methy_ratios': methy_ratios_padded,
            'labels': labels_tensor,
            'batch_labels': batch_labels_tensor,
            "positions": positions_padded,
            "chrs": chrs_padded,
            "prev_is_genome_prev_cpg": prev_is_genome_prev_cpg_padded,
            "file_names": file_names,
        }


class TokensRatiosWithBinsDataset(TokensRatiosDataset):
    """
    继承 TokensRatiosDataset，在返回样本时额外提供:
        - bin_ratios: [bin_dim] 全基因组 sliding window 甲基化比例
        - bin_counts: [bin_dim] 对应 coverage，用于 diffusion loss 加权

    Args 新增:
        bins_root: 存放 *.win{window}_step{step}_ratios.npz 的目录
        window_size, step_size: 生成 bins 时用的参数，用来推断文件名
        bin_dim: (可选) 固定 bins 向量长度；不传则首次 __getitem__ 自动推断
    """

    def __init__(
        self,
        csv_file: str,
        root_path: str,
        K_mer: str,
        tokenizer,
        bins_root: str,
        window_size: int = 100_000,
        step_size: int = 50_000,
        bin_dim: int = None,
        type_json_path: str = None,
        batch_type_json_path: str = None,
        need_labels: bool = False,
        need_batch: bool = False,
        use_length_scale: bool = False,
        bias_power: float = 0.4,
        min_length: int = 50000,
        max_length: int = None,
        random: bool = False,
        use_truncation: bool = False,
        use_sample: bool = False,
        start_idx: int = 0,
        selective_chrs: list = None,
        chrs_random: bool = False,
    ):
        super().__init__(
            csv_file=csv_file,
            root_path=root_path,
            K_mer=K_mer,
            tokenizer=tokenizer,
            type_json_path=type_json_path,
            batch_type_json_path=batch_type_json_path,
            need_labels=need_labels,
            need_batch=need_batch,
            use_length_scale=use_length_scale,
            bias_power=bias_power,
            min_length=min_length,
            max_length=max_length,
            random=random,
            use_truncation=use_truncation,
            use_sample=use_sample,
            start_idx=start_idx,
            selective_chrs=selective_chrs,
            chrs_random=chrs_random,
        )

        self.bins_root = Path(bins_root)
        self.window_size = window_size
        self.step_size = step_size
        self.bin_dim = bin_dim  # 可选，若 None 则第一次 __getitem__ 时根据第一个样本推断

        # 统一的染色体顺序，用于 flatten 全基因组 bins
        # 注意：滑窗脚本里 normalize_chr_id 对 X/Y/M 的编码是：
        #   X -> -1, Y -> 0, M -> -2
        # 保存时 key 类似 ratios_chr1, ratios_chr2, ..., ratios_chr-1, ratios_chr0, ratios_chr-2
        self.chr_order = selective_chrs
        # (
        #     [str(i) for i in range(1, 23)]  # '1' ... '22'
        #     + [-1, 0, -2]                   # X, Y, M 对应的 key
        # )

    def _get_chr_len_and_nbins(self, chr_id, bin_size):
        key = normalize_chr_id(chr_id)
        if key not in chr_len_dict:
            raise KeyError(f"Chromosome {chr_id} (normalized {key}) not in chr_len_dict")
        chr_len = chr_len_dict[key]
        n_bins = int(np.ceil(chr_len / bin_size))
        return key, chr_len, n_bins

    def _load_bins_for_sample(self, file_name: str):
        prefix = Path(file_name).stem
        bins_path = self.bins_root / f"{prefix}.win{self.window_size}_step{self.step_size}_ratios.npz"
        if not bins_path.exists():
            raise FileNotFoundError(f"Bin ratios file not found: {bins_path}")
        bins_npz = np.load(bins_path, allow_pickle=True)

        ratios_keys = [k for k in bins_npz.files if k.startswith("ratios_chr")]
        counts_keys = [k for k in bins_npz.files if k.startswith("counts_chr")]

        chr_bins = []
        for rk, ck in zip(sorted(ratios_keys), sorted(counts_keys)):
            r = bins_npz[rk].astype(np.float32)
            c = bins_npz[ck].astype(np.float32)

            # 记录 NaN 位置
            nan_mask = np.isnan(r)

            # 先填 0，维持shape
            r[nan_mask] = 0.0
            c[nan_mask] = 0.0

            chr_id = rk.replace("ratios_chr", "")
            chr_bins.append((chr_id, r, c, nan_mask))

        if self.chrs_random:
            chrs = [cb[0] for cb in chr_bins]
            unique = list(dict.fromkeys(chrs))
            shuffled = np.random.permutation(unique)
            mapping = {u: s for u, s in zip(unique, shuffled)}
            chr_bins = sorted(chr_bins, key=lambda x: mapping[x[0]])

        # flatten拼接
        bin_ratios  = np.concatenate([b[1] for b in chr_bins], axis=0)
        bin_counts  = np.concatenate([b[2] for b in chr_bins], axis=0)

        # 生成 flatten 对应的 mask
        flat_mask = np.concatenate([b[3] for b in chr_bins], axis=0)

        if self.bin_dim is None:
            self.bin_dim = bin_ratios.shape[0]
        else:
            if bin_ratios.shape[0] != self.bin_dim:
                raise ValueError(f"bin_dim mismatch for {bins_path}")

        return bin_ratios, bin_counts, flat_mask

    def _pad_or_truncate_bins(self, bin_ratios, bin_counts, bin_dim):
        """
        返回：
        - padded_ratios (float32)
        - padded_counts (float32)
        - mask (bool)  True=有效数据  False=NaN或Padding
        """

        # 1) 先记录原始 NaN 位置
        orig_nan_mask = np.isnan(bin_ratios)  # True = NaN 位置

        # 2) 把 NaN 替换成 -1
        bin_ratios = np.nan_to_num(bin_ratios, nan=-1.0).astype(np.float32)
        bin_counts = np.nan_to_num(bin_counts, nan=-1.0).astype(np.float32)

        L = bin_ratios.shape[0]

        # 直接等长，无需pad/truncate
        if L == bin_dim:
            mask = ~orig_nan_mask  # NaN位置为False，其余为True
            return bin_ratios, bin_counts, mask

        # 3) 过长 → 中心截断
        if L > bin_dim:
            start = (L - bin_dim) // 2
            end = start + bin_dim
            # 截断 NaN mask 也要同步中心截断
            trunc_orig_nan_mask = orig_nan_mask[start:end]
            mask = ~trunc_orig_nan_mask  # False = NaN 位置
            return bin_ratios[start:end], bin_counts[start:end], mask

        # 4) 过短 → 右侧padding
        padded_ratios = np.full(bin_dim, -1.0, dtype=np.float32)
        padded_counts = np.full(bin_dim, -1.0, dtype=np.float32)

        padded_ratios[:L] = bin_ratios
        padded_counts[:L] = bin_counts

        # 5) 构建 mask
        mask = np.ones(bin_dim, dtype=bool)  # 先全True
        padded_orig_nan_mask = np.full(bin_dim, True, dtype=bool)  # 默认pad部分都是NaN
        padded_orig_nan_mask[:L] = orig_nan_mask  # 还原前L个的NaN信息
        mask[padded_orig_nan_mask] = False  # 所有 NaN 设为 False
        mask[L:] = False  # pad 位置也设为 False

        return padded_ratios, padded_counts, mask

    def _select_bins_for_region(self, chrs_seq, pos_seq, bins_npz, step_size=50_000):
        """
        chrs_seq:    该 sample 输入的 chrs，长度 = CpG 数（可以是 np.ndarray）
        pos_seq:     该 sample 输入的 positions（0-based，对应 chrs_seq 逐位）
        bins_npz:    np.load 出来的滑窗 npz（某个 prefix.win...npz）
        step_size:   滑窗 step，与预处理一致

        返回：
        bin_ratios_sel: 按 sample["chrs"] 染色体出现顺序拼接后的 bin ratios (1D)
        bin_counts_sel: 对应的 counts (1D)
        bin_idx_sel:    每条染色体的 idx_sel 列表，用于 debug（例如 [(chr_id, idx_array), ...]）
        """
        chrs_seq = np.asarray(chrs_seq)
        pos_seq = np.asarray(pos_seq, dtype=np.int64)

        # ---- 0) 按 chrs_seq 中“首次出现顺序”确定染色体顺序 ----
        # np.unique 默认排序，这里需要保留“出现顺序”
        unique_chrs, first_idx = np.unique(chrs_seq, return_index=True)
        order = np.argsort(first_idx)
        chr_order = unique_chrs[order]  # 按 sample["chrs"] 中的顺序

        all_ratios = []
        all_counts = []
        all_idx_info = []  # [(chr_id, idx_sel_chr), ...]

        # ---- 1) 逐条染色体处理，并按 chr_order 顺序拼接 ----
        for chr_id in chr_order:
            # 当前染色体在这个 sample 中的 CpG 位置
            mask_chr = (chrs_seq == chr_id)
            pos_chr = pos_seq[mask_chr]
            if pos_chr.size == 0:
                continue  # 理论上不会发生，防御性写法

            # 获取该染色体在 bins_npz 中的键名 key（可能是 '1','2' 或 -1,-2,0 等）
            key, chr_len, _ = self._get_chr_len_and_nbins(chr_id, bin_size=step_size)

            # 如果某条 chr 在 bins_npz 里没有，对应键会 KeyError，这里做个安全检查（可选）
            starts_key = f"starts_chr{key}"
            if starts_key not in bins_npz:
                # 可以选择直接跳过，也可以 raise
                # 这里选择跳过，并记录一下，避免训练直接崩
                # print(f"[WARN] chr {chr_id} (key={key}) not found in bins_npz, skip.")
                continue

            starts = bins_npz[f"starts_chr{key}"]  # (n_bins_chr,)
            ends   = bins_npz[f"ends_chr{key}"]
            ratios = bins_npz[f"ratios_chr{key}"]
            counts = bins_npz[f"counts_chr{key}"]

            # ---- 2) 当前染色体 CpG 覆盖区域 ----
            region_start = int(pos_chr.min())
            region_end   = int(pos_chr.max()) + 1  # 右开区间

            # ---- 3) 找到与该区域有交集的 bins ----
            # 条件：窗口 [start_i, end_i) 与 [region_start, region_end) 相交
            mask_bins = (ends > region_start) & (starts < region_end)
            idx_sel_chr = np.nonzero(mask_bins)[0]

            # 如果这一条 chr 没有任何交集，可以选择跳过
            if idx_sel_chr.size == 0:
                # print(f"[INFO] chr {chr_id} no overlapping bins with region [{region_start}, {region_end})")
                continue

            all_ratios.append(ratios[idx_sel_chr])
            all_counts.append(counts[idx_sel_chr])
            all_idx_info.append((chr_id, idx_sel_chr))

        # ---- 4) 把所有染色体结果按顺序拼接 ----
        if len(all_ratios) == 0:
            # 没找到任何 bin（理论上不太可能），返回空数组
            bin_ratios_sel = np.array([], dtype=np.float32)
            bin_counts_sel = np.array([], dtype=np.int64)
        else:
            bin_ratios_sel = np.concatenate(all_ratios, axis=0)
            bin_counts_sel = np.concatenate(all_counts, axis=0)

        return bin_ratios_sel, bin_counts_sel, all_idx_info

    def __getitem__(self, idx: int):
        """
        在原有 sample 基础上，增加:
            - bin_ratios: [bin_dim]
            - bin_counts: [bin_dim]
        """
        # 先用父类生成基础 sample（tokens + ratios + chrs + positions + labels...）
        sample = super().__getitem__(idx)

        file_name = sample["file_names"]  # e.g. something.npy

        # prefix = Path(file_name).stem
        # bin_npz_path = self.bins_root / f"{prefix}.win{self.window_size}_step{self.step_size}_ratios.npz"
        # if not bin_npz_path.exists():
        #     raise FileNotFoundError(f"Bin ratios file not found: {bins_path}")
        # bins_npz = np.load(bin_npz_path, allow_pickle=True)

        # # 2) 从当前 CpG 区域对应的 bins 里拿监督信号 bin_ratios_sel, bin_counts_sel, all_idx_info
        # bin_ratios_sel, bin_counts_sel, _ = self._select_bins_for_region(
        #     chrs_seq=sample["chrs"], pos_seq=sample["positions"],
        #     bins_npz=bins_npz,
        #     step_size=self.step_size,
        # )

        # # 3) 变成固定长度的 bin_dim
        # bin_ratios_fix, bin_counts_fix, flat_masks = self._pad_or_truncate_bins(
        #     bin_ratios_sel, bin_counts_sel, bin_dim=self.bin_dim
        # )

        # 读取该细胞的全基因组 sliding window bins
        bin_ratios, bin_counts, flat_masks = self._load_bins_for_sample(file_name)

        sample["bin_ratios"] = bin_ratios
        sample["bin_counts"] = bin_counts
        sample["flat_masks"] = flat_masks

        return sample
        

@dataclass
class scdnam_collate_TokensRatiosWithBins(scdnam_collate_TokensRatios):
    """
    collate 版本，除了原来的 tokens / ratios / labels 之外，
    额外把 Dataset 提供的:
        - bin_ratios: [bin_dim]
        - bin_counts: [bin_dim]
    拼成 batch:
        - bin_ratios: [B, bin_dim]
        - bin_counts: [B, bin_dim]
    用于 DDIM diffusion loss。
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 先用父类做原来的拼接
        batch = super().__call__(instances)

        # --------- 1. 收集 bin_ratios / bin_counts ---------
        bin_ratios_list = []
        bin_counts_list = []
        flat_masks_list = []

        for item in instances:
            if "bin_ratios" in item:
                bin_ratios_list.append(torch.tensor(item["bin_ratios"], dtype=torch.float))
            if "bin_counts" in item:
                bin_counts_list.append(torch.tensor(item["bin_counts"], dtype=torch.float))
            if "flat_masks" in item:
                flat_masks_list.append(torch.tensor(item["flat_masks"], dtype=torch.bool))

        # --------- 2. stack 成 [B, bin_dim] ---------
        if len(bin_ratios_list) > 0:
            # 假设每个样本的 bin_dim 一样
            bin_ratios = torch.stack(bin_ratios_list, dim=0)  # [B, bin_dim]
        else:
            bin_ratios = None

        if len(bin_counts_list) > 0:
            bin_counts = torch.stack(bin_counts_list, dim=0)  # [B, bin_dim]
        else:
            bin_counts = None

        if len(flat_masks_list) > 0:
            flat_masks = torch.stack(flat_masks_list, dim=0)  # [B, bin_dim]
        else:
            flat_masks = None

        batch["bin_ratios"] = bin_ratios
        batch["bin_counts"] = bin_counts
        batch["flat_masks"] = flat_masks

        return batch
