from __future__ import annotations

import argparse
import ast
import html
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import os

from dotenv import load_dotenv

from load_data import OUTPUT_DIR, find_existing_dataset_root

# Load WANDB_KEY from .env before checking for wandb so login works without
# requiring the user to set the key manually in their shell environment.
load_dotenv()

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

FIELD_SEP = " +++$+++ "
PROCESSED_DIR = Path("data/processed")
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"
PAIRS_JSONL = PROCESSED_DIR / "pairs.jsonl"
STATS_JSON = PROCESSED_DIR / "stats.json"

PROMPT_HEADER = "<prompt> "
RESPONSE_HEADER = " <response> "
MIN_UTTERANCE_CHARS = 1


@dataclass(frozen=True)
class LineRecord:
    """A single line of dialogue from the Cornell corpus."""

    line_id: str
    character_id: str
    movie_id: str
    character_name: str
    text: str


@dataclass(frozen=True)
class DialoguePair:
    """A prompt->response pair extracted from a movie conversation."""

    prompt: str
    response: str
    movie_id: str


def find_corpus_file(root: Path, name: str) -> Path:
    """Recursively search root for a file named name and return the first match.

    Args:
        root: Directory to search under.
        name: Filename to look for.
    Returns:
        Path to the first match found.
    Raises:
        FileNotFoundError: If no matching file is found.
    """
    matches = sorted(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def parse_movie_lines(path: Path) -> dict[str, LineRecord]:
    """Parse movie_lines.txt into a mapping from line ID to LineRecord.

    Args:
        path: Path to movie_lines.txt.
    Returns:
        Dictionary mapping line_id strings to LineRecord instances.
    """
    lines: dict[str, LineRecord] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(FIELD_SEP)
            if len(parts) < 5:
                continue
            line_id, character_id, movie_id, character_name, text = (
                parts[0].strip(),
                parts[1].strip(),
                parts[2].strip(),
                parts[3].strip(),
                FIELD_SEP.join(parts[4:]).strip(),
            )
            lines[line_id] = LineRecord(
                line_id=line_id,
                character_id=character_id,
                movie_id=movie_id,
                character_name=character_name,
                text=clean_text(text),
            )
    return lines


def parse_line_id_list(field: str) -> list[str]:
    """Parse the utterance ID list field from movie_conversations.txt.

    Args:
        field: Raw string field that may be a Python-style list or space-separated IDs.
    Returns:
        List of line ID strings.
    """
    field = field.strip()
    if field.startswith("[") and field.endswith("]"):
        try:
            parsed = ast.literal_eval(field)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            pass
        inner = field[1:-1]
        return [
            token.strip(" \t'\",")
            for token in re.split(r"[\s,]+", inner)
            if token.strip(" \t'\",")
        ]
    return [token for token in field.split() if token]


def parse_movie_conversations(
    path: Path, line_index: dict[str, LineRecord]
) -> list[tuple[str, list[str]]]:
    """Parse movie_conversations.txt into ordered utterance sequences per movie.

    Args:
        path: Path to movie_conversations.txt.
        line_index: Mapping from line_id to LineRecord (from parse_movie_lines).
    Returns:
        List of (movie_id, [utterance_text, ...]) tuples for conversations with >= 2 lines.
    """
    conversations: list[tuple[str, list[str]]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(FIELD_SEP)
            if len(parts) < 4:
                continue
            movie_id = parts[2].strip()
            utterances: list[str] = []
            for line_id in parse_line_id_list(parts[3]):
                record = line_index.get(line_id)
                if record is None:
                    continue
                if record.text:
                    utterances.append(record.text)
            if len(utterances) >= 2:
                conversations.append((movie_id, utterances))
    return conversations


def clean_text(text: str) -> str:
    """Unescape HTML entities, normalize line endings, and collapse whitespace.

    Args:
        text: Raw dialogue text.
    Returns:
        Cleaned text string.
    """
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pairs(
    conversations: list[tuple[str, list[str]]],
    context_turns: int,
) -> list[DialoguePair]:
    """Build prompt->response pairs from conversation utterance sequences.

    Args:
        conversations: List of (movie_id, [utterance, ...]) from parse_movie_conversations.
        context_turns: Number of prior utterances to include as the prompt.
    Returns:
        List of DialoguePair instances.
    Raises:
        ValueError: If context_turns < 1.
    """
    if context_turns < 1:
        raise ValueError("context_turns must be at least 1")

    pairs: list[DialoguePair] = []
    for movie_id, utterances in conversations:
        for idx in range(context_turns, len(utterances)):
            prompt_turns = utterances[idx - context_turns : idx]
            response = utterances[idx]
            prompt = "\n".join(prompt_turns)
            if (
                len(prompt) < MIN_UTTERANCE_CHARS
                or len(response) < MIN_UTTERANCE_CHARS
            ):
                continue
            pairs.append(
                DialoguePair(prompt=prompt, response=response, movie_id=movie_id)
            )
    return pairs


def split_pairs_by_movie(
    pairs: list[DialoguePair],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[DialoguePair], list[DialoguePair], list[DialoguePair]]:
    """Split pairs into train, validation, and test sets at the movie level.

    Splitting by movie prevents dialogue from the same film appearing across
    splits, avoiding data leakage.

    Args:
        pairs: All dialogue pairs to split.
        val_ratio: Fraction of movies to assign to validation (e.g., 0.2 for 60/20/20).
        test_ratio: Fraction of movies to assign to test (e.g., 0.2 for 60/20/20).
        seed: Random seed for reproducible shuffling.
    Returns:
        (train_pairs, val_pairs, test_pairs) tuple.
    Raises:
        ValueError: If fewer than 3 movies are present.
    """
    movie_ids = sorted({pair.movie_id for pair in pairs})
    n = len(movie_ids)
    if n < 3:
        raise ValueError(
            f"Need at least 3 movies for train/val/test split, got {n}."
        )

    rng = random.Random(seed)
    rng.shuffle(movie_ids)

    val_count = max(1, int(n * val_ratio))
    test_count = max(1, int(n * test_ratio))
    while val_count + test_count >= n:
        if test_count > val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break

    test_movies = set(movie_ids[:test_count])
    val_movies = set(movie_ids[test_count : test_count + val_count])

    train_pairs: list[DialoguePair] = []
    val_pairs: list[DialoguePair] = []
    test_pairs: list[DialoguePair] = []
    for pair in pairs:
        if pair.movie_id in test_movies:
            test_pairs.append(pair)
        elif pair.movie_id in val_movies:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)
    return train_pairs, val_pairs, test_pairs


def format_training_text(prompt: str, response: str, eos_token: str) -> str:
    """Format a prompt->response pair as a training string.

    Args:
        prompt: The prompt utterance(s).
        response: The response utterance.
        eos_token: The tokenizer's end-of-sequence token (e.g., '<|endoftext|>').
    Returns:
        Formatted string: '<prompt> {prompt} <response> {response}{eos_token}'
    """
    return f"{PROMPT_HEADER}{prompt}{RESPONSE_HEADER}{response}{eos_token}"


def format_prompt_prefix(prompt: str) -> str:
    """Format the prompt portion only, used for prompt-masking during tokenization.

    Args:
        prompt: The prompt utterance(s).
    Returns:
        Prefix string up to but not including the response content.
    """
    return f"{PROMPT_HEADER}{prompt}{RESPONSE_HEADER}"


def tokenize_pairs(
    pairs: list[DialoguePair],
    tokenizer,
    max_length: int,
    mask_prompt: bool,
) -> list[dict]:
    """Tokenize dialogue pairs into input_ids, attention_mask, and labels.

    Args:
        pairs: DialoguePair instances to tokenize.
        tokenizer: HuggingFace tokenizer (GPT-2).
        max_length: Maximum sequence length; sequences are truncated and padded to this.
        mask_prompt: If True, set prompt token labels to -100 so loss is computed only on response.
    Returns:
        List of dicts with keys: input_ids, attention_mask, labels, movie_id, prompt, response, text.
    """
    eos_token = tokenizer.eos_token or ""
    rows: list[dict] = []
    for pair in pairs:
        full_text = format_training_text(pair.prompt, pair.response, eos_token)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = list(input_ids)

        if mask_prompt:
            prefix = format_prompt_prefix(pair.prompt)
            prefix_ids = tokenizer(
                prefix,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            prompt_len = min(len(prefix_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

        for i, mask in enumerate(attention_mask):
            if mask == 0:
                labels[i] = -100

        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "movie_id": pair.movie_id,
                "prompt": pair.prompt,
                "response": pair.response,
                "text": full_text,
            }
        )
    return rows


def save_jsonl(pairs: list[DialoguePair], path: Path) -> None:
    """Write dialogue pairs to a JSONL file, one JSON object per line.

    Args:
        pairs: DialoguePair instances to serialize.
        path: Destination file path; parent directories are created if needed.
    Side effects:
        Creates or overwrites the file at path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(
                json.dumps(
                    {
                        "prompt": pair.prompt,
                        "response": pair.response,
                        "movie_id": pair.movie_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def compute_stats(
    line_index: dict[str, LineRecord],
    conversations: list[tuple[str, list[str]]],
    train_pairs: list[DialoguePair],
    val_pairs: list[DialoguePair],
    test_pairs: list[DialoguePair],
    tokenized_train: list[dict],
    tokenized_val: list[dict],
    tokenized_test: list[dict],
    args: argparse.Namespace,
) -> dict:
    """Compute summary statistics for the preprocessed dataset.

    Args:
        line_index: Raw line-ID-to-record mapping from parse_movie_lines.
        conversations: Parsed conversation sequences from parse_movie_conversations.
        train_pairs: Training dialogue pairs.
        val_pairs: Validation dialogue pairs.
        test_pairs: Test dialogue pairs.
        tokenized_train: Tokenized training rows.
        tokenized_val: Tokenized validation rows.
        tokenized_test: Tokenized test rows.
        args: Parsed argparse namespace containing preprocessing config values.
    Returns:
        Dictionary of raw counts, pair splits, token-length summaries, and config.
    """
    train_lengths = [len(row["input_ids"]) for row in tokenized_train]
    val_lengths = [len(row["input_ids"]) for row in tokenized_val]
    test_lengths = [len(row["input_ids"]) for row in tokenized_test]
    all_movies = {pair.movie_id for pair in train_pairs + val_pairs + test_pairs}

    def length_summary(lengths: list[int]) -> dict:
        if not lengths:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
        return {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
        }

    return {
        "raw": {
            "utterances": len(line_index),
            "conversations": len(conversations),
            "movies": len(all_movies),
        },
        "pairs": {
            "train": len(train_pairs),
            "validation": len(val_pairs),
            "test": len(test_pairs),
            "total": len(train_pairs) + len(val_pairs) + len(test_pairs),
        },
        "token_lengths": {
            "train": length_summary(train_lengths),
            "validation": length_summary(val_lengths),
            "test": length_summary(test_lengths),
        },
        "config": {
            "context_turns": args.context_turns,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "max_length": args.max_length,
            "mask_prompt": args.mask_prompt,
            "seed": args.seed,
            "model_name": args.model_name,
        },
    }


def resolve_raw_root(raw_dir: Path) -> Path:
    """Locate the Cornell corpus root directory under raw_dir.

    Args:
        raw_dir: Directory where load_data.py placed the downloaded corpus.
    Returns:
        Path to the dataset root containing the Cornell txt files.
    Raises:
        FileNotFoundError: If no Cornell corpus is found under raw_dir.
    """
    root = find_existing_dataset_root(raw_dir)
    if root is None:
        raise FileNotFoundError(
            f"No Cornell corpus found under {raw_dir}. "
            "Run load_data.py first to download the dataset."
        )
    return root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing script.

    Args:
        argv: Argument list to parse; defaults to sys.argv if None.
    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess Cornell Movie Dialogs for GPT-2 fine-tuning."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing the downloaded Kaggle corpus.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory for JSONL pairs, stats, and tokenized datasets.",
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=1,
        help="Number of prior utterances to include in each prompt (1 = single-turn).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of movies reserved for validation (default 0.2 -> 60/20/20 split).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of movies reserved for test (default 0.2 -> 60/20/20 split).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length after tokenization (sequences are padded to this length).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the movie-level train/val split.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="Hugging Face tokenizer model name.",
    )
    parser.add_argument(
        "--mask-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask prompt tokens in labels (-100) so loss applies to the response only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full preprocessing pipeline: parse -> pair -> tokenize -> save.

    Args:
        argv: Argument list; defaults to sys.argv if None.
    Returns:
        Exit code (0 for success, 1 for error).
    Side effects:
        Writes pairs.jsonl, tokenized dataset, and stats.json to output_dir.
    """
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    tokenized_dir = output_dir / "tokenized"
    pairs_jsonl = output_dir / "pairs.jsonl"
    stats_json = output_dir / "stats.json"

    try:
        from datasets import Dataset, DatasetDict
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(
            "Missing dependencies. Install with:\n"
            "  pip install transformers datasets",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    try:
        raw_root = resolve_raw_root(args.raw_dir.resolve())
        lines_path = find_corpus_file(raw_root, "movie_lines.txt")
        conversations_path = find_corpus_file(raw_root, "movie_conversations.txt")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Using raw corpus at: {raw_root}")
    line_index = parse_movie_lines(lines_path)
    conversations = parse_movie_conversations(conversations_path, line_index)
    pairs = build_pairs(conversations, context_turns=args.context_turns)
    try:
        train_pairs, val_pairs, test_pairs = split_pairs_by_movie(
            pairs,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Parsed {len(line_index)} utterances from {lines_path.name}")
    print(f"Built {len(conversations)} conversations with >= 2 resolved lines")
    print(f"Created {len(pairs)} prompt-response pairs")
    print(
        f"Train pairs: {len(train_pairs)} | "
        f"Validation pairs: {len(val_pairs)} | "
        f"Test pairs: {len(test_pairs)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_train = tokenize_pairs(
        train_pairs, tokenizer, args.max_length, args.mask_prompt
    )
    tokenized_val = tokenize_pairs(
        val_pairs, tokenizer, args.max_length, args.mask_prompt
    )
    tokenized_test = tokenize_pairs(
        test_pairs, tokenizer, args.max_length, args.mask_prompt
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_pairs + val_pairs + test_pairs, pairs_jsonl)

    train_ds = Dataset.from_list(tokenized_train)
    val_ds = Dataset.from_list(tokenized_val)
    test_ds = Dataset.from_list(tokenized_test)
    DatasetDict(
        {"train": train_ds, "validation": val_ds, "test": test_ds}
    ).save_to_disk(tokenized_dir)

    stats = compute_stats(
        line_index,
        conversations,
        train_pairs,
        val_pairs,
        test_pairs,
        tokenized_train,
        tokenized_val,
        tokenized_test,
        args,
    )
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Saved pairs to {pairs_jsonl}")
    print(f"Saved tokenized data to {tokenized_dir}")
    print(f"Saved stats to {stats_json}")

    # Log the tokenized dataset as a versioned W&B Artifact. Attaching split
    # sizes and preprocessing config as metadata means any training run that
    # calls use_artifact("cornell-tokenized:latest") creates a traceable
    # data→model lineage edge in the W&B UI without re-uploading the files.
    if _WANDB_AVAILABLE:
        _wandb_key = os.getenv("WANDB_KEY")
        if _wandb_key:
            wandb.login(key=_wandb_key)
        with wandb.init(project="cinechat", job_type="preprocess") as run:
            artifact = wandb.Artifact(
                name="cornell-tokenized",
                type="dataset",
                metadata={
                    "train": len(train_pairs),
                    "validation": len(val_pairs),
                    "test": len(test_pairs),
                    "max_length": args.max_length,
                    "context_turns": args.context_turns,
                    "seed": args.seed,
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                    "model_name": args.model_name,
                },
            )
            artifact.add_dir(str(tokenized_dir))
            run.log_artifact(artifact)
        print("Logged dataset artifact to W&B.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
