from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

DATASET_SLUG = "rajathmc/cornell-moviedialog-corpus"
OUTPUT_DIR = Path("data/raw")
DOWNLOADED_FOLDER = "cornell-moviedialog-corpus"
MARKER_FILE = "movie_lines.txt"
ENV_FILE = Path(".env")


def load_env_file(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def ensure_kaggle_env_vars() -> None:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    api_key = os.getenv("KAGGLE_KEY", "").strip()
    if username and api_key:
        return
    raise RuntimeError(
        "Missing Kaggle credentials in .env. Add KAGGLE_USERNAME and KAGGLE_KEY "
        "to the project .env file, then rerun."
    )


def write_kaggle_json_from_env() -> Path:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    api_key = os.getenv("KAGGLE_KEY", "").strip()
    config_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    config_dir.mkdir(parents=True, exist_ok=True)
    creds_path = config_dir / "kaggle.json"
    creds_path.write_text(
        json.dumps({"username": username, "key": api_key}, indent=2),
        encoding="utf-8",
    )
    return creds_path


def download_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_SLUG} into {OUTPUT_DIR} ...")
    kaggle_api_extended = importlib.import_module("kaggle.api.kaggle_api_extended")
    KaggleApi = kaggle_api_extended.KaggleApi
    api = KaggleApi()
    api.dataset_download_files(DATASET_SLUG, path=str(OUTPUT_DIR), unzip=True)


def count_lines_in_movie_lines(root: Path) -> int:
    marker = find_marker_file(root)
    if marker is None:
        return 0
    return sum(1 for _ in marker.open(encoding="utf-8", errors="replace"))


def find_marker_file(root: Path) -> Path | None:
    for path in root.rglob(MARKER_FILE):
        if path.is_file():
            return path
    return None


def find_existing_dataset_root(base: Path) -> Path | None:
    base = base.resolve()
    for candidate in (base, base / DOWNLOADED_FOLDER):
        if not candidate.is_dir():
            continue
        if find_marker_file(candidate) is not None:
            return candidate
    return None


def main() -> int:
    load_env_file()
    force = "--force" in sys.argv

    out = OUTPUT_DIR.resolve()
    existing = find_existing_dataset_root(out)

    if existing is not None and not force:
        marker = find_marker_file(existing)
        line_count = count_lines_in_movie_lines(existing)
        print(
            f"Dataset already present at {existing} ({MARKER_FILE} found). "
            "Skipping Kaggle download. Use --force to download again."
        )
        print(f"{line_count} lines in {marker}.")
        return 0

    try:
        ensure_kaggle_env_vars()
        creds_path = write_kaggle_json_from_env()
        print(f"Wrote Kaggle credentials file to: {creds_path}")
        download_dataset()
    except RuntimeError as exc:
        print(exc)
        return 1
    except Exception as exc:
        print(f"Dataset download failed: {exc}")
        return 1

    dataset_root = out / DOWNLOADED_FOLDER
    data_root = dataset_root if dataset_root.is_dir() else out
    marker = find_marker_file(data_root)
    line_count = count_lines_in_movie_lines(data_root)
    print(f"Done. Found {line_count} lines in {marker}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())