import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


FOLD_FILENAMES = ("full_dataset_fold1.csv", "full_dataset_fold2.csv")
MERGED_FILENAME = "full_dataset_english_all.csv"


def _read_fold(path):
    return pd.read_csv(
        path,
        sep="\t",
        quotechar='"',
        engine="python",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        keep_default_na=False,
    )


def _write_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _split_patterns(raw_patterns):
    patterns = []
    for raw in raw_patterns:
        patterns.extend(part.strip() for part in str(raw).split(","))
    return [pattern for pattern in patterns if pattern]


def _resolve_excluded_names(available_names, requested_patterns):
    available = sorted(str(name) for name in available_names)
    matched = set()
    unresolved = []

    for pattern in requested_patterns:
        pattern_lower = pattern.lower()
        pattern_slug = _slug(pattern)

        exact_matches = [
            name
            for name in available
            if name.lower() == pattern_lower or _slug(name) == pattern_slug
        ]
        if exact_matches:
            matched.update(exact_matches)
            continue

        loose_matches = [name for name in available if pattern_slug and pattern_slug in _slug(name)]
        if loose_matches:
            matched.update(loose_matches)
        else:
            unresolved.append(pattern)

    if unresolved:
        choices = "\n".join(f"  - {name}" for name in available)
        raise ValueError(
            "Could not match excluded dataset pattern(s): "
            + ", ".join(unresolved)
            + "\nAvailable dataset_of_origin values:\n"
            + choices
        )

    return sorted(matched)


def _load_folds(input_dir):
    input_dir = Path(input_dir)
    folds = {}
    for filename in FOLD_FILENAMES:
        path = input_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing fold file: {path}")
        fold = _read_fold(path)
        if "dataset_of_origin" not in fold.columns:
            raise ValueError(f"{path} is missing required column: dataset_of_origin")
        folds[filename] = fold
    return folds


def list_datasets(input_dir):
    folds = _load_folds(input_dir)
    merged = pd.concat(folds.values(), ignore_index=True)
    counts = merged.groupby("dataset_of_origin").size().sort_values(ascending=False)
    print("dataset_of_origin,num_samples")
    for name, count in counts.items():
        print(f"{name},{count}")


def filter_datasets(input_dir, output_dir, exclude_patterns, dry_run=False):
    folds = _load_folds(input_dir)
    merged_before = pd.concat(folds.values(), ignore_index=True)
    requested_patterns = _split_patterns(exclude_patterns)
    if not requested_patterns:
        raise ValueError("Provide at least one --exclude value.")

    excluded_names = _resolve_excluded_names(
        merged_before["dataset_of_origin"].unique(),
        requested_patterns,
    )

    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Refusing to overwrite --input-dir. Choose a different --output-dir.")

    print("Excluding dataset_of_origin:")
    for name in excluded_names:
        print(f"  - {name}")

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "requested_exclude_patterns": requested_patterns,
        "excluded_dataset_names": excluded_names,
        "folds": {},
    }

    filtered_folds = []
    for filename, fold in folds.items():
        before = len(fold)
        filtered = fold[~fold["dataset_of_origin"].isin(excluded_names)].copy()
        removed = before - len(filtered)
        filtered_folds.append(filtered)
        report["folds"][filename] = {
            "input_rows": int(before),
            "output_rows": int(len(filtered)),
            "removed_rows": int(removed),
        }
        print(f"{filename}: {before} -> {len(filtered)} rows (removed {removed})")

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_tsv(filtered, output_dir / filename)

    merged_after = pd.concat(filtered_folds, ignore_index=True)
    report["merged"] = {
        "input_rows": int(len(merged_before)),
        "output_rows": int(len(merged_after)),
        "removed_rows": int(len(merged_before) - len(merged_after)),
    }
    print(
        f"{MERGED_FILENAME}: {len(merged_before)} -> {len(merged_after)} rows "
        f"(removed {len(merged_before) - len(merged_after)})"
    )

    if not dry_run:
        _write_tsv(merged_after, output_dir / MERGED_FILENAME)
        with open(output_dir / "dataset_filter_report.json", "w") as output_file:
            json.dump(report, output_file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Create a filtered data directory by excluding one or more dataset_of_origin values."
    )
    parser.add_argument("--input-dir", default="data", help="Directory containing full_dataset_fold1/2.csv.")
    parser.add_argument(
        "--output-dir",
        default="data_filtered",
        help="Directory to write filtered fold files.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Dataset name or pattern to exclude. Repeat this option or pass comma-separated values. "
            "Examples: --exclude IEMOCAP --exclude 'fb,Emobank'"
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print available dataset_of_origin values and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without writing files.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        list_datasets(args.input_dir)
        return

    filter_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        exclude_patterns=args.exclude,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
