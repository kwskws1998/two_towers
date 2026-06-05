import argparse
import csv
import os
import zipfile

import pandas as pd


DEFAULT_GDRIVE_ZIP_URL = "https://drive.google.com/file/d/1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj/view?usp=sharing"
DEFAULT_GDRIVE_ZIP_NAME = "english_va_bundle.zip"
DEFAULT_EXTERNAL_DIR = "data/external_english"

EXTERNAL_SOURCE_NAME_MAP = {
    "iemocap": "IEMOCAP sentences",
    "emotales": "EmoTales sentences",
    "scott_et_al": "GlasgowNorms",
    "nrc_vad": "nrc-vad",
    "warriner_et_al": "word ratings ENG",
    "facebook_va": "fb",
    "fb": "fb",
    "emobank": "Emobank",
    "anet": "ANET sentences",
}


def _clean_text_column(series):
    cleaned = series.astype(str)
    cleaned = cleaned.str.replace(r"[\r\n\t]+", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    cleaned = cleaned.str.strip()
    return cleaned


def _normalize_minmax(series):
    series = pd.to_numeric(series, errors="coerce")
    min_value = series.min()
    max_value = series.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return series
    if max_value == min_value:
        return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    normalized = (series - min_value) / (max_value - min_value)
    return normalized.clip(0.0, 1.0)


def _post_process_dataset(df):
    out = df.copy()
    out["text"] = _clean_text_column(out["text"])
    out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
    out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
    out = out.dropna(subset=["text", "valence", "arousal"])
    out = out[out["text"] != ""]

    val_in_unit = out["valence"].between(0.0, 1.0, inclusive="both").all()
    aro_in_unit = out["arousal"].between(0.0, 1.0, inclusive="both").all()
    if val_in_unit and aro_in_unit:
        out["valence"] = out["valence"].clip(0.0, 1.0)
        out["arousal"] = out["arousal"].clip(0.0, 1.0)
    else:
        out["valence"] = _normalize_minmax(out["valence"])
        out["arousal"] = _normalize_minmax(out["arousal"])

    out = out.dropna(subset=["valence", "arousal"])
    out = out.drop_duplicates(subset=["text", "dataset_of_origin"])
    return out


def _download_gdrive_zip(gdrive_url, zip_path, force=False):
    if os.path.isfile(zip_path) and not force:
        print(f"[zip] Already exists, skip download: {zip_path}")
        return zip_path

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download the dataset zip from Google Drive. "
            "Install it with: pip install gdown"
        ) from exc

    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    print(f"[zip] Downloading from Google Drive -> {zip_path}")
    try:
        downloaded = gdown.download(gdrive_url, zip_path, quiet=False, fuzzy=True)
    except TypeError:
        # gdown>=6 removed/changed this parameter; plain URL download still works.
        downloaded = gdown.download(gdrive_url, zip_path, quiet=False)
    if not downloaded or not os.path.isfile(zip_path):
        raise RuntimeError("Failed to download Google Drive dataset zip.")
    return zip_path


def _extract_zip_tsv(zip_path, external_dir, force=False):
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    os.makedirs(external_dir, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            name = member.filename
            base = os.path.basename(name)
            if member.is_dir():
                continue
            if not base.lower().endswith(".tsv"):
                continue
            if name.startswith("__MACOSX/") or base.startswith("._"):
                continue

            out_path = os.path.join(external_dir, base)
            if os.path.isfile(out_path) and not force:
                extracted.append(out_path)
                continue

            with archive.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(out_path)

    if extracted:
        print(f"[zip] TSV files ready in {external_dir}: {len(extracted)}")
    else:
        print(f"[zip] No TSV files extracted from: {zip_path}")
    return extracted


def _infer_dataset_name_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower().replace("-", "_")
    return EXTERNAL_SOURCE_NAME_MAP.get(stem, stem)


def _load_external_sources(external_dir):
    os.makedirs(external_dir, exist_ok=True)
    files = sorted(
        file_name
        for file_name in os.listdir(external_dir)
        if file_name.lower().endswith(".tsv")
    )

    if not files:
        print(f"[external] No TSV files found in: {external_dir}")
        return []

    loaded = []
    for file_name in files:
        path = os.path.join(external_dir, file_name)
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue

        required = {"text", "valence", "arousal"}
        if not required.issubset(set(df.columns)):
            print(f"[warn] Skip {path}: required columns are text, valence, arousal.")
            continue

        dataset_name = _infer_dataset_name_from_path(path)
        out = pd.DataFrame(
            {
                "text": df["text"],
                "valence": df["valence"],
                "arousal": df["arousal"],
                "dataset_of_origin": dataset_name,
            }
        )
        out = _post_process_dataset(out)
        if len(out) == 0:
            print(f"[warn] Skip {path}: no valid rows after processing.")
            continue

        loaded.append(out)
        print(f"[external] Loaded {file_name} -> {dataset_name}: {len(out)} rows")

    return loaded


def _split_in_two_folds(df, seed):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled.insert(0, "index", shuffled.index.astype(int))
    midpoint = len(shuffled) // 2
    fold1 = shuffled.iloc[:midpoint].copy()
    fold2 = shuffled.iloc[midpoint:].copy()
    return fold1, fold2


def _write_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
        escapechar="\\",
    )


def build_english_dataset(
    output_dir,
    seed,
    force=False,
    external_dir=DEFAULT_EXTERNAL_DIR,
    gdrive_zip_url=DEFAULT_GDRIVE_ZIP_URL,
    gdrive_zip_name=DEFAULT_GDRIVE_ZIP_NAME,
    skip_gdrive_download=False,
):
    fold1_path = os.path.join(output_dir, "full_dataset_fold1.csv")
    fold2_path = os.path.join(output_dir, "full_dataset_fold2.csv")
    merged_path = os.path.join(output_dir, "full_dataset_english_all.csv")

    if (
        not force
        and os.path.isfile(fold1_path)
        and os.path.isfile(fold2_path)
        and os.path.isfile(merged_path)
    ):
        print("English dataset already exists. Skipping download/build.")
        print(f"Use --force to rebuild: {fold1_path}, {fold2_path}")
        return

    os.makedirs(external_dir, exist_ok=True)
    zip_path = os.path.join(external_dir, gdrive_zip_name)
    if not skip_gdrive_download:
        _download_gdrive_zip(gdrive_zip_url, zip_path, force=force)
    if os.path.isfile(zip_path):
        _extract_zip_tsv(zip_path, external_dir, force=force)

    dataframes = _load_external_sources(external_dir)
    if not dataframes:
        raise RuntimeError(
            "No valid dataset TSV files available. "
            f"Put TSV files in {external_dir} or provide a valid --gdrive-zip-url."
        )

    merged = pd.concat(dataframes, ignore_index=True)
    merged = merged[["text", "dataset_of_origin", "valence", "arousal"]]
    merged = merged.drop_duplicates(subset=["text", "dataset_of_origin"])

    fold1, fold2 = _split_in_two_folds(merged, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    _write_tsv(fold1, fold1_path)
    _write_tsv(fold2, fold2_path)
    _write_tsv(pd.concat([fold1, fold2], ignore_index=True), merged_path)

    counts = merged.groupby("dataset_of_origin").size().sort_values(ascending=False)
    print("English dataset prepared.")
    print(f"Total samples: {len(merged)}")
    print("Samples per source:")
    for name, value in counts.items():
        print(f"- {name}: {value}")
    print(f"Saved: {fold1_path}")
    print(f"Saved: {fold2_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download (Google Drive) and build English-only VA folds from TSV files."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write full_dataset_fold1.csv and full_dataset_fold2.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed used before splitting into fold1/fold2",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild files even if full_dataset_fold1/2 and full_dataset_english_all already exist.",
    )
    parser.add_argument(
        "--external-dir",
        default=DEFAULT_EXTERNAL_DIR,
        help="Folder containing dataset TSV files (text,valence,arousal).",
    )
    parser.add_argument(
        "--gdrive-zip-url",
        default=DEFAULT_GDRIVE_ZIP_URL,
        help="Google Drive share URL for the dataset zip.",
    )
    parser.add_argument(
        "--gdrive-zip-name",
        default=DEFAULT_GDRIVE_ZIP_NAME,
        help="Filename to store downloaded zip under --external-dir.",
    )
    parser.add_argument(
        "--skip-gdrive-download",
        action="store_true",
        help="Skip gdown download and use already existing TSV files in --external-dir.",
    )
    args = parser.parse_args()

    build_english_dataset(
        output_dir=args.output_dir,
        seed=args.seed,
        force=args.force,
        external_dir=args.external_dir,
        gdrive_zip_url=args.gdrive_zip_url,
        gdrive_zip_name=args.gdrive_zip_name,
        skip_gdrive_download=args.skip_gdrive_download,
    )


if __name__ == "__main__":
    main()
