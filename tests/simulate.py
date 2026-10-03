import sys
sys.path.insert(0, ".")
import pandas as pd
from src.chat_engine import Engine
from src import knowledge as K

eng = Engine()
test = pd.read_csv("data/processed/test.csv")
names = dict(pd.read_csv("data/processed/label_mapping.csv").values)

first_ok = final_ok = asked_total = skipped = 0
for text, truth in zip(test.processed_text, test.disease):
    ctx = None
    r = eng.respond(text, ctx)
    ctx = r["context"]
    p0 = eng.probabilities(ctx)
    if p0 is None:
        skipped += 1
        continue
    first_ok += eng.classes[p0.argmax()] == truth
    turns = 0
    while not any(x["type"] == "result" for x in r["replies"]) and turns < 8:
        q = next(x for x in r["replies"] if x["type"] == "question")
        pend = ctx["pending"]
        if pend == "location":
            msg = "skip"
        else:
            msg = "yes" if truth in K.QUESTIONS[pend]["supports"] else "no"
        r = eng.respond(msg, ctx)
        ctx = r["context"]
        turns += 1
    asked_total += turns
    res = next(x for x in r["replies"] if x["type"] == "result")
    final_ok += res["disease"]["name"] == K.DISEASES[truth]["name"]

n = len(test)
print(f"skipped={skipped} ")
print(f"n={n}  first-pass top1={first_ok/n:.3f}  after follow-ups={final_ok/n:.3f}  avg questions={asked_total/n:.2f}")
