"""Generate every figure used in the README from the current data and model.

Run from the project root:  python src/make_figures.py
Figures are written to docs/figures/.
"""

import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, log_loss)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, ".")
from src import knowledge as K  # noqa: E402

OUT = "docs/figures"
GREEN, DARK, GREY, AMBER, RED = "#2c6b57", "#16241c", "#9aa59d", "#c9893a", "#a3342b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
raw = pd.read_csv("data/raw/skin_disease_dataset.csv")
model = pickle.load(open("models/skin_disease_model.pkl", "rb"))
names = pickle.load(open("models/label_names.pkl", "rb"))
classes = [names[i] for i in model.classes_]
pretty = [c.replace("_", " ") for c in classes]
Xtr, ytr = train.processed_text.values, train.label.values
Xte, yte = test.processed_text.values, test.label.values
stats = {}


def save(name):
    os.makedirs(OUT, exist_ok=True)
    plt.savefig(os.path.join(OUT, name))
    plt.close()
    print("wrote", name)


def tfidf():
    return TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)


# 1. class distribution --------------------------------------------------------
def fig_distribution():
    counts = raw.groupby(["disease", "source"]).size().unstack(fill_value=0)
    counts = counts.loc[classes]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(pretty, counts.get("synthetic_seed", 0), color=GREY, label="hand-written seed")
    ax.bar(pretty, counts.get("synthetic_llm", 0), bottom=counts.get("synthetic_seed", 0), color=GREEN, label="generated realistic")
    ax.set_ylabel("messages")
    ax.set_title(f"Dataset: {len(raw):,} messages, balanced across 10 classes")
    ax.legend(frameon=False, loc="lower right")
    plt.xticks(rotation=35, ha="right")
    save("01_class_distribution.png")


# 2. message length ------------------------------------------------------------
def fig_length():
    n = raw.text.str.split().str.len()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(n.clip(upper=45), bins=range(1, 47, 2), color=GREEN, edgecolor="white")
    ax.axvline(n.median(), color=AMBER, ls="--")
    ax.text(n.median() + 0.8, ax.get_ylim()[1] * 0.9, f"median {int(n.median())} words", color=AMBER)
    ax.set_xlabel("words per message (capped at 45)")
    ax.set_ylabel("messages")
    ax.set_title("Message length: from short fragments to long stories")
    save("02_message_length.png")


# 3. model comparison ----------------------------------------------------------
def fig_models():
    from sklearn.model_selection import cross_val_score
    cands = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": LinearSVC(max_iter=10000),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    rows = []
    for n, clf in cands.items():
        p = Pipeline([("t", tfidf()), ("c", clf)])
        cv = cross_val_score(p, Xtr, ytr, cv=5).mean()
        p.fit(Xtr, ytr)
        rows.append((n, cv, accuracy_score(yte, p.predict(Xte))))
    stats["models"] = rows
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, color=GREY, label="5-fold CV")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, color=GREEN, label="test")
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r[2] + 0.01, f"{r[2]:.1%}", ha="center", fontsize=9)
        ax.text(i - 0.2, r[1] + 0.01, f"{r[1]:.1%}", ha="center", fontsize=9, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("accuracy")
    ax.set_title("Four classifiers on the same TF-IDF features")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    save("03_model_comparison.png")


# 4. confusion matrix ----------------------------------------------------------
def fig_confusion():
    pred = model.predict(Xte)
    cm = confusion_matrix(yte, pred)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in ((axes[0], cm, "Counts", "d"), (axes[1], cmn, "Row-normalized (recall)", ".0%")):
        im = ax.imshow(data, cmap="Greens")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(pretty, rotation=45, ha="right")
        ax.set_yticklabels(pretty)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix: {title}")
        for i in range(10):
            for j in range(10):
                v = data[i, j]
                if v > 0 and (fmt == "d" or v >= 0.01):
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                            color="white" if v > data.max() * 0.55 else DARK)
    plt.tight_layout()
    save("04_confusion_matrix.png")
    off = [(cm[i, j], pretty[i], pretty[j]) for i in range(10) for j in range(10) if i != j]
    stats["top_confusions"] = sorted(off, reverse=True)[:5]


# 5. per-class metrics ---------------------------------------------------------
def fig_per_class():
    rep = classification_report(yte, model.predict(Xte), target_names=classes, output_dict=True)
    order = sorted(classes, key=lambda c: rep[c]["f1-score"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(order))
    for k, (m, col) in enumerate((("precision", GREY), ("recall", AMBER), ("f1-score", GREEN))):
        ax.barh(y + (k - 1) * 0.26, [rep[c][m] for c in order], 0.26, color=col, label=m)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in order])
    ax.set_xlim(0.5, 1.02)
    ax.set_title("Per-class precision, recall and F1 (test set)")
    ax.legend(frameon=False, loc="lower right")
    save("05_per_class_metrics.png")
    stats["report"] = {c: rep[c] for c in classes}


# 6. epoch-by-epoch training curve --------------------------------------------
def fig_epochs():
    """The saved model uses the L-BFGS solver (no epochs). This curve trains the same
    model family (logistic regression) with SGD so convergence per epoch can be shown.
    A validation slice is held out of the training set; the test set is never touched."""
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.15, stratify=ytr, random_state=1)
    vec = tfidf().fit(Xa)
    A, V = vec.transform(Xa), vec.transform(Xv)
    clf = SGDClassifier(loss="log_loss", alpha=1 / (10 * A.shape[0]), learning_rate="constant",
                        eta0=0.5, random_state=0)
    rng = np.random.RandomState(0)
    hist = {"tl": [], "vl": [], "ta": [], "va": []}
    cls = np.unique(ytr)
    for ep in range(1, 61):
        idx = rng.permutation(A.shape[0])
        clf.partial_fit(A[idx], ya[idx], classes=cls)
        hist["tl"].append(log_loss(ya, clf.predict_proba(A), labels=cls))
        hist["vl"].append(log_loss(yv, clf.predict_proba(V), labels=cls))
        hist["ta"].append(accuracy_score(ya, clf.predict(A)))
        hist["va"].append(accuracy_score(yv, clf.predict(V)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    e = range(1, 61)
    axes[0].plot(e, hist["tl"], color=GREEN, label="train loss")
    axes[0].plot(e, hist["vl"], color=AMBER, label="validation loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("log loss")
    axes[0].set_title("Loss per epoch")
    axes[0].legend(frameon=False)
    axes[1].plot(e, hist["ta"], color=GREEN, label="train accuracy")
    axes[1].plot(e, hist["va"], color=AMBER, label="validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy per epoch")
    final = hist["vl"][-1]
    plateau = next(i for i, v in enumerate(hist["vl"], 1) if v <= final * 1.01)
    axes[0].axvline(plateau, color=GREY, ls=":")
    axes[0].text(plateau + 1, max(hist["tl"]) * 0.8, f"validation loss\nflat from epoch {plateau}", color="#555", fontsize=9)
    axes[1].legend(frameon=False, loc="center right")
    best = plateau
    plt.tight_layout()
    save("06_training_epochs.png")
    stats["epochs"] = {"plateau_epoch": best, "val_acc_final": hist["va"][-1], "train_acc_final": hist["ta"][-1],
                       "val_loss_min": min(hist["vl"])}


# 7. learning curve ------------------------------------------------------------
def fig_learning_curve():
    p = Pipeline([("t", tfidf()), ("c", LogisticRegression(max_iter=2000, C=10))])
    sizes, tr, va = learning_curve(p, Xtr, ytr, cv=5, train_sizes=np.linspace(0.1, 1.0, 8), random_state=0, n_jobs=1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for arr, col, lab in ((tr, GREEN, "training"), (va, AMBER, "cross-validation")):
        ax.plot(sizes, arr.mean(1), "-o", color=col, label=lab, ms=4)
        ax.fill_between(sizes, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), color=col, alpha=0.15)
    ax.set_xlabel("training messages")
    ax.set_ylabel("accuracy")
    ax.set_title("Learning curve: more data still helps")
    ax.legend(frameon=False, loc="lower right")
    save("07_learning_curve.png")
    stats["learning"] = {"n_max": int(sizes[-1]), "val_at_10pct": float(va.mean(1)[0]), "val_at_100pct": float(va.mean(1)[-1])}


# 8. top-k accuracy ------------------------------------------------------------
def fig_topk():
    p = model.predict_proba(Xte)
    ks = [1, 2, 3, 5]
    vals = [np.mean([yte[i] in model.classes_[np.argsort(p[i])[-k:]] for i in range(len(yte))]) for k in ks]
    stats["topk"] = dict(zip(ks, vals))
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar([f"top {k}" for k in ks], vals, color=[GREY, GREEN, GREEN, GREEN])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center")
    ax.set_ylim(0.6, 1.05)
    ax.set_ylabel("correct answer within top k")
    ax.set_title("Right answer is almost always in the top 3")
    save("08_topk_accuracy.png")


# 9. calibration ---------------------------------------------------------------
def fig_calibration():
    p = model.predict_proba(Xte)
    conf, ok = p.max(1), (p.argmax(1) == model.classes_.searchsorted(yte)).astype(float) if False else (model.classes_[p.argmax(1)] == yte).astype(float)
    bins = np.linspace(0.1, 1.0, 10)
    idx = np.digitize(conf, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 5:
            xs.append(conf[m].mean())
            ys.append(ok[m].mean())
            ns.append(int(m.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].plot([0, 1], [0, 1], ls="--", color=GREY, label="perfect calibration")
    axes[0].plot(xs, ys, "-o", color=GREEN, label="model (C = 10)")
    axes[0].set_xlabel("stated confidence (top probability)")
    axes[0].set_ylabel("actual accuracy")
    axes[0].set_title("Reliability: does confidence mean anything?")
    axes[0].legend(frameon=False)
    axes[1].hist(conf, bins=15, color=GREEN, edgecolor="white")
    axes[1].axvline(0.70, color=AMBER, ls="--")
    axes[1].text(0.71, axes[1].get_ylim()[1] * 0.9, "ask below 0.70", color=AMBER)
    axes[1].set_xlabel("top probability")
    axes[1].set_ylabel("messages")
    axes[1].set_title("Confidence distribution")
    plt.tight_layout()
    save("09_calibration.png")
    stats["calib"] = {"mean_conf": float(conf.mean()), "acc": float(ok.mean())}


# 10. regularization sweep -----------------------------------------------------
def fig_c_sweep():
    Cs = [1, 3, 10, 30, 100]
    accs, confs = [], []
    for C in Cs:
        p = Pipeline([("t", tfidf()), ("c", LogisticRegression(max_iter=3000, C=C))]).fit(Xtr, ytr)
        pr = p.predict_proba(Xte)
        accs.append(accuracy_score(yte, p.predict(Xte)))
        confs.append(pr.max(1).mean())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(Cs, accs, "-o", color=GREEN, label="test accuracy")
    ax.plot(Cs, confs, "-s", color=AMBER, label="average stated confidence")
    ax.set_xscale("log")
    ax.axvline(10, color=GREY, ls=":")
    ax.set_xlabel("regularization C (log scale)")
    ax.set_title("Accuracy stays flat; stated confidence rises with C")
    ax.legend(frameon=False, loc="center left")
    save("10_regularization_sweep.png")


# 11. top words per class ------------------------------------------------------
def fig_top_words():
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feats = vec.get_feature_names_out()
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.2))
    for ax, i in zip(axes.ravel(), range(10)):
        top = np.argsort(clf.coef_[i])[-7:]
        ax.barh([feats[j] for j in top], clf.coef_[i][top], color=GREEN)
        ax.set_title(pretty[i], fontsize=10)
        ax.tick_params(labelsize=8)
    plt.suptitle("Words with the strongest weight for each disease", fontweight="bold", y=1.02)
    plt.tight_layout()
    save("11_top_words.png")


# 12. follow-up questions effect ----------------------------------------------
def fig_followups():
    from src.chat_engine import Engine
    eng = Engine()
    rows = []
    for text, truth in zip(test.processed_text, test.disease):
        r = eng.respond(text, None)
        ctx = r["context"]
        p0 = eng.probabilities(ctx)
        if p0 is None:
            continue
        before = eng.classes[p0.argmax()] == truth
        turns = 0
        while not any(x["type"] == "result" for x in r["replies"]) and turns < 8:
            pend = ctx["pending"]
            msg = "skip" if pend == "location" else ("yes" if truth in K.QUESTIONS[pend]["supports"] else "no")
            r = eng.respond(msg, ctx)
            ctx = r["context"]
            turns += 1
        res = next(x for x in r["replies"] if x["type"] == "result")
        rows.append((truth, before, res["disease"]["name"] == K.DISEASES[truth]["name"], turns))
    df = pd.DataFrame(rows, columns=["disease", "before", "after", "turns"])
    g = df.groupby("disease")[["before", "after"]].mean().loc[classes]
    stats["followup"] = {"before": float(df.before.mean()), "after": float(df.after.mean()),
                         "avg_q": float(df.turns.mean()), "n": len(df), "dist": df.turns.value_counts().sort_index().to_dict()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(10)
    axes[0].bar(x - 0.2, g.before, 0.4, color=GREY, label=f"first answer ({df.before.mean():.1%})")
    axes[0].bar(x + 0.2, g.after, 0.4, color=GREEN, label=f"after follow-ups ({df.after.mean():.1%})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pretty, rotation=40, ha="right")
    axes[0].set_ylim(0.4, 1.2)
    axes[0].set_ylabel("accuracy")
    axes[0].set_title("Follow-up questions help every disease (simulated)")
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    d = df.turns.value_counts().sort_index()
    axes[1].bar(d.index.astype(str), d.values, color=GREEN)
    axes[1].set_xlabel("questions asked")
    axes[1].set_ylabel("conversations")
    axes[1].set_title(f"Questions per chat (avg {df.turns.mean():.2f})")
    plt.tight_layout()
    save("12_followup_effect.png")


def main():
    fig_distribution()
    fig_length()
    fig_models()
    fig_confusion()
    fig_per_class()
    fig_epochs()
    fig_learning_curve()
    fig_topk()
    fig_calibration()
    fig_c_sweep()
    fig_top_words()
    fig_followups()
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    print(json.dumps(stats, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
