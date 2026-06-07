from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import DialogueDataset
from model import (
    CHECKPOINTS_DIR,
    DEFAULT_MODEL_NAME,
    get_device,
    load_checkpoint,
    load_model,
    load_tokenizer,
)

# Load WANDB_KEY (and any other vars) from .env before wandb is imported.
# This must happen before the wandb import so the key is available when
# wandb initializes its internal state at import time.
load_dotenv()

# wandb is optional, all logging calls are guarded by wandb.run is not None
# so the script runs fine without it, just without experiment tracking.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Fixed prompts used at the end of every epoch to qualitatively track how
# generation quality changes over training. Keeping them identical across
# epochs and runs makes the W&B Table easy to compare side-by-side.
SAMPLE_PROMPTS = [
    "You can't handle the truth.",
    "I'm going to make him an offer he can't refuse.",
    "Why so serious?",
    "I'll be back.",
    "Elementary, my dear Watson.",
]

# Random hyperparameter search over LR, batch size, and warmup steps.
# Random search is preferred over grid search because of compute
# 6 random samples give better per-parameter coverage at a fraction of the cost.
SWEEP_CONFIG = {
    "method": "random",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [2e-5, 5e-5, 1e-4]},
        "batch_size": {"values": [4, 8, 16]},
        "warmup_steps": {"values": [50, 100, 200]},
    },
    "run_cap": 6,
}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    # Shared evaluation logic used by both train() (for validation) and test().
    # Runs the model in eval mode over the full loader and returns
    # (mean_loss, perplexity). Perplexity = exp(mean_loss).
    # The model is returned to train mode before this function exits so the
    # caller's training loop is unaffected.
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Passing labels to GPT-2 triggers internal cross-entropy loss
            # computation. Positions masked to -100 by preprocess.py (prompt
            # tokens and padding) are automatically excluded from the loss.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

    mean_loss = total_loss / n_batches if n_batches > 0 else float("inf")

    # Guard against OverflowError when loss is very high early in training.
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = float("inf")

    model.train()
    return mean_loss, perplexity


def generate_samples(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 60,
) -> list[dict[str, str]]:
    # Generate one response per fixed prompt using the same prompt/response
    # format that preprocess.py used during tokenization. The model needs to see
    # those exact tags to understand the prompt/response structure it was trained on.
    # Returns a list of dicts shaped for direct insertion into a wandb Table.
    model.eval()
    samples = []

    with torch.no_grad():
        for prompt_text in SAMPLE_PROMPTS:
            formatted = f"<prompt> {prompt_text} <response> "
            input_ids = tokenizer(
                formatted,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            # Slice off the prompt token IDs so only the generated response
            # is decoded. Without this, the prompt text would appear in the output.
            generated = tokenizer.decode(
                output_ids[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            samples.append({
                "prompt": prompt_text,
                "generated_response": generated.strip(),
            })

    model.train()
    return samples


def train(args: argparse.Namespace) -> float:
    # Full training loop with per-batch W&B metric logging and per-epoch
    # validation, checkpointing, artifact upload, and sample generation.
    # Returns the best validation loss achieved across all epochs.

    device = get_device()
    print(f"Training on device: {device}")

    # Initialize W&B for a standalone run. If a sweep agent already called
    # wandb.init(), wandb.run is not None and we skip this block, instead
    # pulling hyperparams from wandb.config so the sweep controls the values.
    if _WANDB_AVAILABLE:
        # Authenticate using the key from .env. wandb.login() is a no-op if
        # the key is already valid, so this is safe to call every run.
        _wandb_key = os.getenv("WANDB_KEY")
        if _wandb_key:
            wandb.login(key=_wandb_key)
        if wandb.run is None:
            wandb.init(
                project="cinechat",
                name=f"run-lr{args.learning_rate}-bs{args.batch_size}",
                config={
                    "model": args.model_name,
                    "dataset": "cornell-movie-dialogs",
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "num_epochs": args.num_epochs,
                    "max_length": 128,
                    "warmup_steps": args.warmup_steps,
                    "weight_decay": args.weight_decay,
                },
            )
        else:
            # Sweep agent initialized wandb, override args with swept values.
            cfg = wandb.config
            args.learning_rate = getattr(cfg, "learning_rate", args.learning_rate)
            args.batch_size = getattr(cfg, "batch_size", args.batch_size)
            args.warmup_steps = getattr(cfg, "warmup_steps", args.warmup_steps)

        # Declare the dataset artifact this run consumed so W&B draws the
        # data->model lineage edge. Wrapped in try/except so a missing artifact
        # (e.g. preprocess.py not yet re-run) doesn't abort training.
        if wandb.run is not None:
            try:
                wandb.run.use_artifact("cornell-tokenized:latest")
            except Exception:
                pass

    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name, device)
    model.train()

    # pin_memory=True speeds up host-to-GPU data transfers on CUDA machines.
    pin = device.type == "cuda"

    train_loader = DataLoader(
        DialogueDataset("train"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        DialogueDataset("validation"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Linear warmup then linear decay to zero over the full training run.
    # total_steps tells the scheduler when to finish the decay.
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # clip_grad_norm_ clips in-place AND returns the pre-clip norm.
            # Logging the pre-clip norm lets us spot gradient explosions early,
            # a norm that consistently hits the ceiling (1.0) signals instability.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            epoch_train_loss += loss.item()

            # Per-step logging builds the fine-grained loss, LR, and grad_norm
            # curves in W&B. Step-level granularity is important for spotting
            # spikes or instability that epoch averages would hide.
            if _WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                })

        # Compute epoch-level train perplexity from the mean batch loss.
        mean_train_loss = epoch_train_loss / len(train_loader)
        try:
            train_perplexity = math.exp(mean_train_loss)
        except OverflowError:
            train_perplexity = float("inf")

        val_loss, val_perplexity = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"train loss: {mean_train_loss:.4f} | train ppl: {train_perplexity:.2f} | "
            f"val loss: {val_loss:.4f} | val ppl: {val_perplexity:.2f}"
        )

        # Epoch-level perplexity points form the perplexity curves in W&B.
        # These are separate from the per-step loss log above.
        if _WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "train/perplexity": train_perplexity,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "epoch": epoch,
            })

        # Save a per-epoch checkpoint in HuggingFace format (config.json +
        # pytorch_model.bin). This lets any epoch be reloaded for comparison.
        epoch_ckpt_dir = CHECKPOINTS_DIR / f"epoch-{epoch}"
        model.save_pretrained(epoch_ckpt_dir)
        tokenizer.save_pretrained(epoch_ckpt_dir)

        # Log each checkpoint as a versioned W&B Artifact. Storing val_loss
        # in metadata means the best checkpoint is identifiable from the W&B UI
        # without re-running evaluation.
        if _WANDB_AVAILABLE and wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"cinechat-epoch-{epoch}",
                type="model",
                metadata={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": vars(args),
                },
            )
            artifact.add_dir(str(epoch_ckpt_dir))
            wandb.log_artifact(artifact)

        # Overwrite the "best" checkpoint whenever val loss improves.
        # test() always loads from this directory so it doesn't need to know
        # which epoch was best.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_dir = CHECKPOINTS_DIR / "best"
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"  → New best val loss {best_val_loss:.4f}, saved to {best_ckpt_dir}")

        # Generate sample dialogues and log as a W&B Table. One row per prompt
        # per epoch, so you can scroll through the table and watch response
        # quality improve (or degrade) over time.
        samples = generate_samples(model, tokenizer, device)
        if _WANDB_AVAILABLE and wandb.run is not None:
            table = wandb.Table(columns=["prompt", "generated_response", "epoch"])
            for s in samples:
                table.add_data(s["prompt"], s["generated_response"], epoch)
            wandb.log({"samples": table, "epoch": epoch})

    return best_val_loss


def test(args: argparse.Namespace) -> tuple[float, float]:
    # Load the best checkpoint written by train() and evaluate on the held-out
    # test split. Logs test/loss and test/perplexity to W&B if a run is active.
    # The test set is intentionally kept separate from train() so it is only
    # evaluated once, after all training and hyperparameter decisions are final.

    device = get_device()
    tokenizer = load_tokenizer(args.model_name)

    best_ckpt = CHECKPOINTS_DIR / "best"
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"No best checkpoint found at {best_ckpt}. Run train() first."
        )

    model = load_checkpoint(best_ckpt, args.model_name, device)

    pin = device.type == "cuda"
    test_loader = DataLoader(
        DialogueDataset("test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    test_loss, test_perplexity = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f} | Test perplexity: {test_perplexity:.2f}")

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"test/loss": test_loss, "test/perplexity": test_perplexity})

    return test_loss, test_perplexity


def run_sweep(args: argparse.Namespace) -> None:
    # Initialize a W&B sweep and launch the agent. The agent calls _sweep_run()
    # up to run_cap times, each with a different sampled hyperparameter config.
    if not _WANDB_AVAILABLE:
        print("wandb not installed. Run: pip install wandb", file=sys.stderr)
        return

    sweep_id = wandb.sweep(SWEEP_CONFIG, project="cinechat")

    def _sweep_run() -> None:
        # The sweep agent calls this once per trial. wandb.init() here gives
        # the trial its own run. train() detects the active run and pulls
        # hyperparams from wandb.config instead of args.
        with wandb.init():
            train(args)

    wandb.agent(sweep_id, function=_sweep_run, count=SWEEP_CONFIG["run_cap"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CineChat: GPT-2 fine-tuned on Cornell Movie Dialogs."
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a W&B hyperparameter sweep instead of a single training run.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Skip training and evaluate the best saved checkpoint on the test set.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.test_only:
        try:
            test(args)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0

    if args.sweep:
        run_sweep(args)
        return 0

    # Default path: train then immediately evaluate on the test set.
    train(args)
    try:
        test(args)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
