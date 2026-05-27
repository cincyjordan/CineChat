from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model import CHECKPOINTS_DIR, DEFAULT_MODEL_NAME, get_device, load_checkpoint, load_tokenizer

# These tags must match preprocess.py exactly. The model learned to produce
# a response after seeing <response>, so the inference prefix must use the
# same format or the model won't know where to start generating.
PROMPT_HEADER = "<prompt> "
RESPONSE_HEADER = " <response> "


def load_for_inference(
    checkpoint_path: Path | str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> tuple:
    # Load model and tokenizer from a checkpoint directory, ready for inference.
    # Defaults to checkpoints/best, the directory train.py keeps updated with
    # the lowest val loss checkpoint seen so far.
    # Returns (model, tokenizer, device).
    if device is None:
        device = get_device()
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "best"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run train.py first to produce a checkpoint."
        )

    # load_checkpoint already puts the model in eval() mode.
    tokenizer = load_tokenizer(model_name)
    model = load_checkpoint(checkpoint_path, model_name, device)
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> str:
    # Format the user's prompt with the training tags and run a single
    # forward pass through the model to generate a response.
    # No gradients are computed, this is pure inference.
    formatted = f"{PROMPT_HEADER}{prompt_text}{RESPONSE_HEADER}"
    encoding = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoding.input_ids.to(device)
    # Explicitly pass the attention mask to avoid GPT-2's pad==eos ambiguity.
    # Without this, the model can't distinguish padding from end-of-sequence
    # and produces garbled output.
    attention_mask = encoding.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens before decoding so the returned string
    # contains only the newly generated response text.
    # Strip U+FFFD and U+0120 (Ġ), GPT-2's BPE space prefix token that
    # some terminals render as a replacement character at the start of output.
    response = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )
    return response.replace('�', '').replace('Ġ', '').strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a movie dialogue-style response from a fine-tuned GPT-2 model."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input line of dialogue to respond to.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint directory (defaults to checkpoints/best).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        model, tokenizer, device = load_for_inference(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    response = generate_response(
        model,
        tokenizer,
        args.prompt,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(f"Prompt  : {args.prompt}")
    print(f"Response: {response}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
