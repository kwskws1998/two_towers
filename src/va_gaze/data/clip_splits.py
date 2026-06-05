import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_FILENAMES = {
    "train": "train.tsv",
    "val": "val.tsv",
    "test": "test.tsv",
    "holdout": "iemocap_zeroshot.tsv",
}


def read_vad_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        quotechar='"',
        engine="python",
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
        keep_default_na=False,
    )


def write_vad_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def slug(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def load_merged_english_data(data_dir):
    data_dir = Path(data_dir)
    merged_path = data_dir / "full_dataset_english_all.csv"
    if merged_path.is_file():
        return read_vad_tsv(merged_path)

    fold_paths = [data_dir / "full_dataset_fold1.csv", data_dir / "full_dataset_fold2.csv"]
    if all(path.is_file() for path in fold_paths):
        folds = [read_vad_tsv(path) for path in fold_paths]
        return pd.concat(folds, ignore_index=True)

    raise FileNotFoundError(
        f"Could not find {merged_path} or full_dataset_fold1/2.csv under {data_dir}. "
        "Run prepare_english_data.py first."
    )


def resolve_holdout_names(df, holdout_pattern):
    pattern_slug = slug(holdout_pattern)
    matched = []
    for name in sorted(df["dataset_of_origin"].dropna().unique()):
        name_slug = slug(name)
        if pattern_slug == name_slug or pattern_slug in name_slug:
            matched.append(name)
    if not matched:
        available = "\n".join(f"  - {name}" for name in sorted(df["dataset_of_origin"].unique()))
        raise ValueError(
            f"Could not match holdout dataset pattern: {holdout_pattern}\n"
            f"Available dataset_of_origin values:\n{available}"
        )
    return matched


def _can_stratify(values, test_size):
    counts = pd.Series(values).value_counts()
    if len(counts) < 2:
        return False
    if counts.min() < 2:
        return False
    expected_test_min = int(np.floor(float(test_size) * counts.min()))
    return expected_test_min >= 1


def _split_train_val_test(df, seed, val_size, test_size):
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1.")

    temp_size = val_size + test_size
    stratify_first = df["dataset_of_origin"] if _can_stratify(df["dataset_of_origin"], temp_size) else None
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify_first,
    )

    relative_test = test_size / temp_size
    stratify_second = (
        temp_df["dataset_of_origin"]
        if _can_stratify(temp_df["dataset_of_origin"], relative_test)
        else None
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=stratify_second,
    )
    return train_df, val_df, test_df


def build_no_iemocap_splits(
    data_dir,
    output_dir,
    holdout_dataset="IEMOCAP",
    seed=42,
    val_size=0.1,
    test_size=0.1,
    force=False,
):
    output_dir = Path(output_dir)
    split_paths = {name: output_dir / filename for name, filename in SPLIT_FILENAMES.items()}
    if not force and all(path.is_file() for path in split_paths.values()):
        return split_paths

    df = load_merged_english_data(data_dir)
    required = {"index", "text", "dataset_of_origin", "valence", "arousal"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in English VA data: {sorted(missing)}")

    holdout_names = resolve_holdout_names(df, holdout_dataset)
    holdout_df = df[df["dataset_of_origin"].isin(holdout_names)].copy()
    train_pool = df[~df["dataset_of_origin"].isin(holdout_names)].copy()
    if len(holdout_df) == 0:
        raise ValueError(f"No holdout rows found for pattern: {holdout_dataset}")
    if len(train_pool) < 10:
        raise ValueError("Not enough no-IEMOCAP rows to split into train/val/test.")

    train_df, val_df, test_df = _split_train_val_test(
        train_pool,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "holdout": holdout_df,
    }
    for name, split_df in split_dfs.items():
        split_df = split_df.sort_values("index").reset_index(drop=True)
        write_vad_tsv(split_df, split_paths[name])

    report = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "holdout_dataset_pattern": holdout_dataset,
        "holdout_dataset_names": holdout_names,
        "seed": seed,
        "val_size": val_size,
        "test_size": test_size,
        "num_rows": {name: int(len(split_df)) for name, split_df in split_dfs.items()},
        "train_pool_dataset_counts": train_pool.groupby("dataset_of_origin").size().to_dict(),
        "holdout_dataset_counts": holdout_df.groupby("dataset_of_origin").size().to_dict(),
    }
    with open(output_dir / "split_report.json", "w") as output_file:
        json.dump(report, output_file, indent=2)

    return split_paths
