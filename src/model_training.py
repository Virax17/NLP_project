"""
Skin Disease Chatbot - Model Training Pipeline
================================================
Trains a TF-IDF + classifier pipeline on the preprocessed dataset.
Evaluates multiple classifiers and saves the best one.

DISCLAIMER: This project is for academic purposes only.
It does not provide medical diagnoses.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import cross_val_score


SOURCES_TEST = None


def load_splits(processed_dir: str):
    global SOURCES_TEST
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))
    label_map = pd.read_csv(os.path.join(processed_dir, "label_mapping.csv"))

    X_train = train_df["processed_text"].values
    y_train = train_df["label"].values
    X_test = test_df["processed_text"].values
    y_test = test_df["label"].values

    label_names = dict(zip(label_map["label"], label_map["disease"]))
    if "source" in test_df.columns:
        SOURCES_TEST = test_df["source"].values

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes: {list(label_names.values())}")
    return X_train, y_train, X_test, y_test, label_names


def build_pipelines():
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LinearSVC(max_iter=10000, C=1.0)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]),
    }
    return pipelines


def evaluate_all(pipelines, X_train, y_train, X_test, y_test, label_names):
    results = {}

    for name, pipeline in pipelines.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {acc:.4f}")
        if SOURCES_TEST is not None:
            for src in sorted(set(SOURCES_TEST)):
                m = SOURCES_TEST == src
                print(f"  {src}: {accuracy_score(y_test[m], y_pred[m]):.4f} (n={int(m.sum())})")

        target_names = [label_names[i] for i in sorted(label_names.keys())]
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        results[name] = {
            "pipeline": pipeline,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": acc,
            "y_pred": y_pred,
        }

    return results


def select_best(results):
    # The chatbot needs class probabilities to decide when to ask follow-ups,
    # so only models with predict_proba are eligible.
    eligible = {
        k: v for k, v in results.items()
        if hasattr(v["pipeline"].named_steps["clf"], "predict_proba")
    }
    best_name = max(eligible, key=lambda k: eligible[k]["test_accuracy"])
    best = eligible[best_name]
    print(f"\nBest model: {best_name}")
    print(f"  CV accuracy:   {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})")
    print(f"  Test accuracy: {best['test_accuracy']:.4f}")
    return best_name, best["pipeline"]


def word_purity(X_train, y_train):
    """For each word: (share of its documents in the most common class, document count).
    The chatbot uses this to tell condition-specific words from generic ones."""
    cv = CountVectorizer(binary=True)
    M = cv.fit_transform(X_train)
    out = {}
    classes = np.unique(y_train)
    per_class = np.vstack([np.asarray(M[y_train == c].sum(axis=0)).ravel() for c in classes])
    total = per_class.sum(axis=0)
    for w, i in cv.vocabulary_.items():
        out[w] = (float(per_class[:, i].max() / total[i]), int(total[i]))
    return out


def save_model(pipeline, label_names, purity, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "skin_disease_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    with open(os.path.join(model_dir, "word_purity.pkl"), "wb") as f:
        pickle.dump(purity, f)

    labels_path = os.path.join(model_dir, "label_names.pkl")
    with open(labels_path, "wb") as f:
        pickle.dump(label_names, f)

    print(f"\nModel saved to {model_path}")
    print(f"Labels saved to {labels_path}")
    return model_path


def predict_symptom(text: str, model_path="models/skin_disease_model.pkl",
                    labels_path="models/label_names.pkl"):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    with open(labels_path, "rb") as f:
        label_names = pickle.load(f)

    prediction = pipeline.predict([text])[0]
    disease = label_names[prediction]

    print(f"\nInput: {text}")
    print(f"Predicted disease: {disease}")
    print("\nDISCLAIMER: This is for informational purposes only.")
    print("Please consult a dermatologist for proper diagnosis.")
    return disease


def main():
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models"

    X_train, y_train, X_test, y_test, label_names = load_splits(PROCESSED_DIR)

    pipelines = build_pipelines()
    results = evaluate_all(pipelines, X_train, y_train, X_test, y_test, label_names)

    best_name, best_pipeline = select_best(results)
    save_model(best_pipeline, label_names, word_purity(X_train, y_train), MODEL_DIR)

    print("\n" + "=" * 50)
    print("Testing with sample inputs")
    print("=" * 50)

    test_inputs = [
        "itchy red patches on my elbows",
        "small pimples and oily skin on face",
        "circular ring shaped rash on arm",
        "white patches losing color on hands",
        "red face with visible blood vessels",
        "itchy bumps between fingers at night",
    ]

    for text in test_inputs:
        predict_symptom(text, f"{MODEL_DIR}/skin_disease_model.pkl",
                       f"{MODEL_DIR}/label_names.pkl")


if __name__ == "__main__":
    main()
