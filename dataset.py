from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Root directory for all preprocessed outputs written by preprocess.py.
PROCESSED_DIR = Path("data/processed")

# Subdirectory containing the HuggingFace DatasetDict (train/validation/test splits).
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"

VALID_SPLITS = ("train", "validation", "test")

# Only these three fields are returned as tensors. The remaining fields in each
# row (prompt, response, movie_id, text) are string metadata that the DataLoader
# cannot stack into a batch, so they are intentionally excluded.
TENSOR_KEYS = ("input_ids", "attention_mask", "labels")


class DialogueDataset(Dataset):
    # PyTorch Dataset wrapping the tokenized Cornell Movie Dialogs corpus.
    # Reads the HuggingFace DatasetDict saved by preprocess.py and serves
    # input_ids, attention_mask, and labels as LongTensors to a DataLoader.
    # All sequences are already padded to a fixed length by preprocess.py,
    # so no custom collate_fn is needed, PyTorch's default collator handles
    # stacking uniform-length tensors automatically.

    def __init__(self, split: str, tokenized_dir: Path = TOKENIZED_DIR) -> None:
        # Fail early with a clear message if preprocess.py has not been run yet.
        if not tokenized_dir.exists():
            raise FileNotFoundError(
                f"Tokenized dataset not found at {tokenized_dir}. "
                "Run preprocess.py first."
            )

        # Import deferred so the module can be imported even without datasets
        # installed, surfacing the error only when actually constructing the dataset.
        from datasets import load_from_disk

        dataset_dict = load_from_disk(str(tokenized_dir))

        # Guard against typos in the split name before anything else runs.
        if split not in dataset_dict:
            raise ValueError(
                f"Split '{split}' not found in dataset at {tokenized_dir}. "
                f"Available splits: {list(dataset_dict.keys())}"
            )

        self._data = dataset_dict[split]
        self._split = split

    @property
    def split(self) -> str:
        # Expose the split name so callers (e.g. train.py logging) can read it
        # without needing to track it separately.
        return self._split

    def __len__(self) -> int:
        # Required by PyTorch's DataLoader to determine batch boundaries.
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Convert the three numeric fields from Python lists (how HuggingFace
        # stores them on disk) into LongTensors. torch.long matches the dtype
        # expected by GPT-2's embedding layer and the cross-entropy loss.
        row = self._data[idx]
        return {key: torch.tensor(row[key], dtype=torch.long) for key in TENSOR_KEYS}


if __name__ == "__main__":
    # sload all three splits and print counts, shapes, and dtypes.
    # Exits with code 1 if preprocess.py has not been run yet.
    for split in VALID_SPLITS:
        try:
            ds = DialogueDataset(split)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue

        sample = ds[0]
        print(
            f"{split:>12} | {len(ds):>7} examples | "
            f"input_ids shape: {sample['input_ids'].shape} | "
            f"dtype: {sample['input_ids'].dtype}"
        )
