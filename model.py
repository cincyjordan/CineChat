from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default model name. GPT-2 Small has 124M parameters across 12 transformer
# decoder layers, 12 attention heads, and 768-dimensional embeddings.
# Using the string gpt2 loads the smallest variant from the HuggingFace hub.
DEFAULT_MODEL_NAME = "gpt2"

# Directory where train.py saves model checkpoints after each epoch.
# Kept as a Path constant so all scripts will reference the same location.
CHECKPOINTS_DIR = Path("checkpoints")


def get_device() -> torch.device:
    # Resolve the best available compute device at runtime.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> AutoTokenizer:
    # Load the GPT-2 BPE tokenizer from the HuggingFace hub.
    # GPT-2 was not trained with a dedicated pad token, so pad_token is None
    # by default. Assigning eos_token as the pad token is the standard fix,
    # it matches what preprocess.py does and keeps tokenizer behavior consistent
    # between preprocessing and inference.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> AutoModelForCausalLM:
    # Load pretrained GPT-2 weights for causal language modeling.
    # AutoModelForCausalLM is used instead of GPT2LMHeadModel directly
    # this keeps the code forward-compatible if the model name ever changes
    # to a different causal LM (e.g. gpt2-medium, distilgpt2).
    # Full fine-tuning: all 124M parameters remain unfrozen and are updated
    # during training.
    # The model is moved to the target device immediately after loading so
    # the caller never has to think about device placement.
    if device is None:
        device = get_device()

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    return model


def load_checkpoint(
    checkpoint_path: Path | str,
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> AutoModelForCausalLM:
    # Load a fine-tuned checkpoint saved by train.py for inference.
    # Supports two checkpoint formats:
    # 1. A directory written by model.save_pretrained(): the HuggingFace
    # standard format that stores config.json + pytorch_model.bin.
    # 2. A .pt file written by torch.save(model.state_dict(), path):
    # loads the state dict into a fresh pretrained model.
    # After loading, the model is put into eval() mode and moved to device.
    # eval() disables dropout layers and batch norm running stats updates,
    # which is required for deterministic inference behavior.
    if device is None:
        device = get_device()

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train.py first to produce a checkpoint."
        )

    if checkpoint_path.is_dir():
        # HuggingFace save_pretrained format, config and weights are bundled
        # in the directory so no base model_name is needed.
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    else:
        # Raw state dict, initialize from pretrained weights first so that
        # any keys not saved (e.g. buffers) fall back to their pretrained values.
        model = AutoModelForCausalLM.from_pretrained(model_name)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: AutoModelForCausalLM) -> dict[str, int]:
    # Count total and trainable parameters in the model.
    # Useful for confirming all layers are unfrozen before training starts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # load the tokenizer and model, print architecture summary
    # and parameter counts. Exits with code 1 on any failure so CI can catch it.
    device = get_device()
    print(f"Device: {device}")

    print("Loading tokenizer...")
    try:
        tokenizer = load_tokenizer()
    except Exception as exc:
        print(f"Failed to load tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Vocab size : {tokenizer.vocab_size}")
    print(f"Pad token  : {tokenizer.pad_token!r}")
    print(f"EOS token  : {tokenizer.eos_token!r}")

    print("\nLoading model...")
    try:
        model = load_model(device=device)
    except Exception as exc:
        print(f"Failed to load model: {exc}", file=sys.stderr)
        sys.exit(1)

    params = count_parameters(model)
    print(f"Total parameters    : {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"\nModel config:\n{model.config}")
