"""Merge the hand-written seed set with the generated realistic messages,
clean them and write one raw CSV for data_preprocessing.py.

Run from the project root:  python src/build_dataset.py
"""

import csv
import glob
import re

import pandas as pd

SEED = "data/raw/skin_disease_symptoms.csv"
GENERATED = "data/raw/generated/*.csv"
OUT = "data/raw/skin_disease_dataset.csv"

LABELS = {
    "acne", "eczema", "psoriasis", "ringworm", "vitiligo", "rosacea",
    "contact_dermatitis", "hives", "seborrheic_dermatitis", "scabies",
}


def key(text):
    """Normalized form used to catch exact and near duplicates."""
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return " ".join(words)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    seed = pd.read_csv(SEED)
    seed["source"] = "synthetic_seed"
    frames = [seed]
    for path in sorted(glob.glob(GENERATED)):
        df = pd.read_csv(path, quoting=csv.QUOTE_MINIMAL)
        frames.append(df)
        print(f"{path}: {len(df)} rows")

    df = pd.concat(frames, ignore_index=True)
    for col in ("symptoms", "body_location", "severity"):
        if col not in df:
            df[col] = ""
    df = df[["text", "disease", "symptoms", "body_location", "severity", "source"]]

    df["text"] = df["text"].astype(str).str.replace("—", "-").str.replace("–", "-").str.strip()
    df["disease"] = df["disease"].astype(str).str.strip().str.lower()
    df = df[df["disease"].isin(LABELS) & (df["text"].str.len() >= 8)]

    df["_key"] = df["text"].map(key)
    before = len(df)
    df = df.drop_duplicates(subset="_key")
    print(f"exact duplicates removed: {before - len(df)}")

    # Near-duplicate removal inside each class (token-set Jaccard >= 0.8).
    keep = []
    for label, grp in df.groupby("disease"):
        seen = []
        for idx, k in zip(grp.index, grp["_key"]):
            toks = set(k.split())
            if any(jaccard(toks, s) >= 0.8 for s in seen):
                continue
            seen.append(toks)
            keep.append(idx)
    removed = len(df) - len(keep)
    df = df.loc[keep].drop(columns="_key")
    print(f"near duplicates removed: {removed}")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}: {len(df)} rows")
    print(df["disease"].value_counts().to_string())
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
