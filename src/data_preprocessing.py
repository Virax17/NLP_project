"""
Skin Disease Chatbot - Data Preprocessing Pipeline
===================================================
This script loads the raw symptom dataset, cleans it,
preprocesses the text, and saves a training-ready CSV.

DISCLAIMER: This project is for academic purposes only.
It does not provide medical diagnoses.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ── 1. Load the raw CSV ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    return df


# ── 2. Initial exploration ───────────────────────────────────────────

def explore_data(df: pd.DataFrame) -> None:
    print("\n-- Data Overview --")
    print(df.info())

    print("\n-- First 5 rows --")
    print(df.head())

    print("\n-- Missing values --")
    print(df.isnull().sum())

    print("\n-- Disease distribution --")
    print(df["disease"].value_counts())

    print(f"\n-- Unique diseases: {df['disease'].nunique()} --")

    print("\n-- Text length stats --")
    df["_text_len"] = df["text"].astype(str).apply(len)
    print(df["_text_len"].describe())
    df.drop(columns=["_text_len"], inplace=True)


# ── 3. Remove duplicates ────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["text", "disease"])
    after = len(df)
    print(f"Removed {before - after} duplicate rows ({before} -> {after})")
    return df.reset_index(drop=True)


# ── 4. Handle missing values ────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "disease"])
    df["text"] = df["text"].astype(str).str.strip()
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    print(f"After removing missing/empty: {len(df)} rows")
    return df.reset_index(drop=True)


# ── 5. Standardize disease labels ───────────────────────────────────

LABEL_MAP = {
    "acne": "acne",
    "eczema": "eczema",
    "atopic dermatitis": "eczema",
    "psoriasis": "psoriasis",
    "ringworm": "ringworm",
    "tinea": "ringworm",
    "vitiligo": "vitiligo",
    "rosacea": "rosacea",
    "contact dermatitis": "contact_dermatitis",
    "contact_dermatitis": "contact_dermatitis",
    "hives": "hives",
    "urticaria": "hives",
    "seborrheic dermatitis": "seborrheic_dermatitis",
    "seborrheic_dermatitis": "seborrheic_dermatitis",
    "dandruff": "seborrheic_dermatitis",
    "scabies": "scabies",
}


def standardize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["disease"] = df["disease"].str.lower().str.strip()
    df["disease"] = df["disease"].map(LABEL_MAP).fillna(df["disease"])

    known = set(LABEL_MAP.values())
    unknown = df[~df["disease"].isin(known)]
    if len(unknown) > 0:
        print(f"WARNING: {len(unknown)} rows with unknown labels: {unknown['disease'].unique()}")
        df = df[df["disease"].isin(known)]

    print(f"Standardized labels. Distribution:\n{df['disease'].value_counts()}")
    return df.reset_index(drop=True)


# ── 6. Clean text ───────────────────────────────────────────────────
#
# What we DO:
#   - lowercase
#   - remove URLs and emails
#   - remove extra whitespace
#   - keep hyphens in medical terms (e.g. "pus-filled")
#   - keep numbers (duration/count can matter: "3 days", "2 weeks")
#
# What we do NOT remove:
#   - negation words ("not", "no", "without") — medically critical
#   - body part words — needed for context
#   - severity words ("mild", "severe") — informative
#   - color words ("red", "white", "yellow") — key symptom descriptors

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 7. Tokenization and Lemmatization ───────────────────────────────
#
# We use a CONSERVATIVE stopword list: we keep negation words and
# medically relevant common words that standard stopword lists remove.

KEEP_WORDS = {
    "not", "no", "nor", "never", "without",
    "very", "more", "most", "much",
    "few", "all", "both", "each", "every",
    "after", "before", "during", "between",
    "under", "over", "above", "below",
}


def get_safe_stopwords() -> set:
    stops = set(stopwords.words("english"))
    stops -= KEEP_WORDS
    return stops


def tokenize_and_lemmatize(text: str, stop_words: set, lemmatizer) -> str:
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── 8. Remove short / invalid entries ────────────────────────────────

def remove_short_texts(df: pd.DataFrame, min_words: int = 2) -> pd.DataFrame:
    before = len(df)
    df["_word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))
    df = df[df["_word_count"] >= min_words]
    df.drop(columns=["_word_count"], inplace=True)
    print(f"Removed {before - len(df)} entries with fewer than {min_words} words")
    return df.reset_index(drop=True)


# ── 9. Encode labels ────────────────────────────────────────────────

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["disease"])
    print(f"\nLabel encoding:")
    for cls, idx in zip(le.classes_, le.transform(le.classes_)):
        count = (df["label"] == idx).sum()
        print(f"  {idx}: {cls} ({count} samples)")
    return df, le


# ── 10. Train / test split ──────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ── Main pipeline ───────────────────────────────────────────────────

def main():
    RAW_PATH = "data/raw/skin_disease_dataset.csv"
    PROCESSED_DIR = "data/processed"

    # Load
    df = load_data(RAW_PATH)
    explore_data(df)

    # Clean
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = standardize_labels(df)

    # Text preprocessing
    df["cleaned_text"] = df["text"].apply(clean_text)

    stop_words = get_safe_stopwords()
    lemmatizer = WordNetLemmatizer()
    df["processed_text"] = df["cleaned_text"].apply(
        lambda t: tokenize_and_lemmatize(t, stop_words, lemmatizer)
    )

    df = remove_short_texts(df, min_words=2)

    # Show raw vs processed examples
    print("\n-- Raw vs Processed Examples --")
    for i in range(min(5, len(df))):
        print(f"  Raw:       {df.iloc[i]['text']}")
        print(f"  Processed: {df.iloc[i]['processed_text']}")
        print(f"  Disease:   {df.iloc[i]['disease']}")
        print()

    # Encode
    df, label_encoder = encode_labels(df)

    # Save full processed dataset
    output_cols = ["processed_text", "disease", "label", "source"]
    df[output_cols].to_csv(f"{PROCESSED_DIR}/dataset_processed.csv", index=False)

    # Also save a simple two-column version for quick use
    df[["processed_text", "disease"]].rename(
        columns={"processed_text": "text"}
    ).to_csv(f"{PROCESSED_DIR}/dataset_simple.csv", index=False)

    # Split and save
    train_df, test_df = split_data(df)
    train_df[output_cols].to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df[output_cols].to_csv(f"{PROCESSED_DIR}/test.csv", index=False)

    # Save label mapping
    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    pd.DataFrame(list(label_map.items()), columns=["disease", "label"]).to_csv(
        f"{PROCESSED_DIR}/label_mapping.csv", index=False
    )

    print(f"\nAll files saved to {PROCESSED_DIR}/")
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
