"""Conversation engine for the skin chatbot.

Turns free text into one of: small talk, a disease question, an off-topic
reply, or a symptom description. Symptoms accumulate across turns. When the
model is not confident the engine asks a targeted follow-up question, then
folds the answer back into the prediction.

Informational only. Not a medical diagnosis tool.
"""

import difflib
import os
import pickle
import re

import nltk
import numpy as np

for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src import knowledge as K

KEEP_WORDS = {
    "not", "no", "nor", "never", "without", "very", "more", "most", "much",
    "few", "all", "both", "each", "every", "after", "before", "during",
    "between", "under", "over", "above", "below",
}

DISCLAIMER = "This is for information only. Please see a dermatologist for a proper diagnosis."
ASK_BELOW = 0.70
MIN_MARGIN = 0.30
START_BUDGET = 4
SPECIFIC_PURITY = 0.6
SPECIFIC_MIN_DOCS = 6
YES_BOOST = 2.5
NO_PENALTY = 0.35

NEG_RE = re.compile(
    r"\b(?:no|not|without|never|nor)\s+(?:(?:very|really|so|too|that|much|any|longer)\s+)?\w+"
)
SKIN_RE = re.compile(r"\b(?:" + K.SKIN_TERMS + r")")
BODY_RE = re.compile(r"\b(?:" + K.BODY_PARTS + r")")

YES_RE = re.compile(
    r"^(y|yes|yeah|yep|yup|yea|sure|correct|right|absolutely|definitely|true|i do|it does|they do|"
    r"it is|there is|there are|i have|a little|sometimes|somewhat|kind of|mostly|ya)\b"
)
NO_RE = re.compile(
    r"^(n|no|nope|nah|not really|never|none|not at all|i do not|it does not|they do not|it is not|"
    r"there is not|negative)\b"
)
UNSURE_RE = re.compile(
    r"\b(not sure|do not know|dont know|maybe|unsure|idk|cannot tell|no idea|skip|unknown|perhaps|"
    r"rather not|prefer not|pass)\b"
)

GREET_RE = re.compile(r"^(hi+|hello+|hey+|yo|hola|namaste|good (morning|afternoon|evening))\b")
THANKS_RE = re.compile(r"\b(thanks|thank you|thx|thankyou|appreciate|cheers)\b")
BYE_RE = re.compile(r"^(bye|goodbye|see you|that is all|that's all|exit|quit|good night)\b")
HOWARE_RE = re.compile(r"how are you|how is it going|how's it going|what's up|whats up")
IDENT_RE = re.compile(r"who are you|what are you|your name|are you (a |an )?(real |human|bot|robot|ai|doctor|chatgpt)|who made you|who built you")
HELP_RE = re.compile(r"what can you do|how (do|does) (this|it|you) work|how to use|what do you do|^help\b|can you help")
OK_RE = re.compile(r"^(ok|okay|k|fine|cool|alright|got it|i see|hmm+|great|nice|good|understood)\b")
RESET_RE = re.compile(r"\b(start over|new chat|reset|restart|clear (the )?(chat|conversation)|begin again)\b")
JUST_TELL_RE = re.compile(r"\b(just tell me|answer now|no more questions|enough questions|stop asking|give me (the )?(answer|result)|tell me now)\b")
MOLE_RE = re.compile(r"\b(mole|melanoma|skin cancer|cancer|tumou?r|carcinoma)\b")
CHILD_RE = re.compile(r"\b(child|kid|baby|toddler|infant|son|daughter)\b")

ASPECTS = [
    ("contagious", re.compile(r"contagious|spread|catch|infectious|pass (it )?on|transmit|give (it )?to")),
    ("treatment", re.compile(r"\b(treat|treatment|cure|heal|medicine|medication|cream|ointment|lotion|remed|get rid|home remedy|manage|what should i do|what can i do|how (to|do i|can i) (stop|fix|get rid))")),
    ("causes", re.compile(r"\b(cause|causes|why (do|did|am|does|is)|reason|trigger|triggers|due to|how did i get|how do you get)\b")),
    ("see_doctor", re.compile(r"\b(doctor|dermatologist|serious|dangerous|worry|worried|emergency|hospital|urgent|see a|should i go|visit)\b")),
    ("symptoms", re.compile(r"\b(symptom|symptoms|signs?|look like|looks like)\b")),
    ("prevent", re.compile(r"\b(prevent|avoid|come back|recur|stop it from|keep it from)\b")),
    ("what_is", re.compile(r"\b(what is|what's|what are|tell me about|explain|define|meaning of|info on|information (on|about)|about)\b")),
]

QUESTION_CUE = re.compile(r"^(how|what|why|which|can|could|should|is|are|does|do|will|when|where|who|any|tell me|explain)")

ASPECT_TITLES = {
    "contagious": "Is it contagious?",
    "treatment": "How is it usually managed?",
    "causes": "What causes it?",
    "see_doctor": "When to see a doctor",
    "symptoms": "Common signs",
    "prevent": "Keeping it under control",
    "what_is": "About this condition",
}

IGNORED_ALIASES = {"dandruff", "welts", "mites", "pimple problem", "allergic rash", "allergic reaction rash"}


def _alias_pattern(alias):
    return re.compile(r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b")


class Engine:
    def __init__(self, model_dir="models"):
        with open(os.path.join(model_dir, "skin_disease_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, "label_names.pkl"), "rb") as f:
            names = pickle.load(f)
        self.classes = [names[i] for i in self.model.classes_]
        self.index = {c: i for i, c in enumerate(self.classes)}
        self.tfidf = self.model.named_steps["tfidf"]
        vocab = list(self.tfidf.vocabulary_)
        self.unigrams = sorted(w for w in vocab if " " not in w)
        self.unigram_set = set(self.unigrams)
        # A word is "specific" when most training messages containing it share one disease.
        with open(os.path.join(model_dir, "word_purity.pkl"), "rb") as f:
            purity = pickle.load(f)
        self.specific = {w for w, (p, n) in purity.items() if p >= SPECIFIC_PURITY and n >= SPECIFIC_MIN_DOCS}
        self.stop = set(stopwords.words("english")) - KEEP_WORDS
        self.lem = WordNetLemmatizer()
        self.alias_patterns = []
        for key, d in K.DISEASES.items():
            for alias in d["aliases"]:
                if alias not in IGNORED_ALIASES:
                    self.alias_patterns.append((key, _alias_pattern(alias)))

    # ---------- text preparation ----------

    def prep(self, text):
        t = text.lower().replace("’", "'")
        t = t.replace("won't", "will not").replace("can't", "cannot")
        t = re.sub(r"n't\b", " not", t)
        t = NEG_RE.sub(" ", t)
        t = re.sub(r"[^a-z0-9\s\-]", " ", t)
        toks = []
        for w in t.split():
            toks.extend(K.SYNONYMS.get(w, w).split())
        toks = [self.lem.lemmatize(w) for w in toks if w not in self.stop and len(w) > 1]
        out = []
        for w in toks:
            if w not in self.unigram_set and len(w) >= 4:
                m = difflib.get_close_matches(w, self.unigrams, n=1, cutoff=0.8)
                if m and m[0][0] == w[0]:
                    w = m[0]
            out.append(w)
        return " ".join(out)

    def probabilities(self, ctx):
        text = self.prep(" ".join(ctx["symptoms"]))
        if self.tfidf.transform([text]).nnz == 0:
            return None
        p = self.model.predict_proba([text])[0].copy()
        for d, factor in ctx["boosts"].items():
            p[self.index[d]] *= factor
        return p / p.sum()

    # ---------- context ----------

    @staticmethod
    def new_context():
        return {"symptoms": [], "boosts": {}, "asked": [], "pending": None,
                "last": None, "budget": START_BUDGET, "answered": False}

    def clean_context(self, raw):
        ctx = self.new_context()
        if not isinstance(raw, dict):
            return ctx
        syms = raw.get("symptoms")
        if isinstance(syms, list):
            ctx["symptoms"] = [str(s)[:400] for s in syms[-12:]]
        boosts = raw.get("boosts")
        if isinstance(boosts, dict):
            for k, v in boosts.items():
                if k in self.index and isinstance(v, (int, float)):
                    ctx["boosts"][k] = float(min(max(v, 0.01), 50.0))
        asked = raw.get("asked")
        if isinstance(asked, list):
            ctx["asked"] = [a for a in asked if a == "location" or a in K.QUESTIONS][:20]
        pend = raw.get("pending")
        if pend == "location" or pend in K.QUESTIONS:
            ctx["pending"] = pend
        if raw.get("last") in self.index:
            ctx["last"] = raw["last"]
        b = raw.get("budget")
        if isinstance(b, int):
            ctx["budget"] = min(max(b, 0), 30)
        ctx["answered"] = bool(raw.get("answered"))
        return ctx

    # ---------- reply builders ----------

    @staticmethod
    def text(msg):
        return {"type": "text", "text": msg}

    def examples_quick(self):
        return ["itchy red patches on my elbows", "small pimples and oily skin on my face",
                "circular rash on my arm", "white patches where my skin lost color"]

    def after_answer_quick(self):
        return ["Is it contagious?", "How is it treated?", "When should I see a doctor?", "Start over"]

    # ---------- main entry ----------

    def respond(self, message, raw_ctx):
        ctx = self.clean_context(raw_ctx)
        msg = " ".join(str(message).split())[:400]
        low = msg.lower().replace("’", "'")
        replies, quick = [], []

        if not msg:
            return self.out([self.text("Type a message and I'll do my best to help.")], [], ctx)

        if any(re.search(p, low) for p in K.RED_FLAGS):
            replies.append({"type": "urgent", "text": K.URGENT_TEXT})

        if RESET_RE.search(low):
            ctx = self.new_context()
            return self.out(replies + [self.text("Okay, starting fresh. Describe what you see or feel on your skin.")],
                            self.examples_quick(), ctx)

        if ctx["pending"]:
            handled = self.handle_pending(msg, low, ctx)
            if handled is not None:
                r, q = handled
                return self.out(replies + r, q, ctx)

        words = low.split()
        has_skin = bool(SKIN_RE.search(low)) or bool(SKIN_RE.search(self.prep(msg)))
        found = self.find_diseases(low)
        aspects = self.find_aspects(low)

        if JUST_TELL_RE.search(low) and ctx["symptoms"]:
            r, q = self.answer(ctx, intro="Understood, here is my best read with what I have.")
            return self.out(replies + r, q, ctx)

        if MOLE_RE.search(low):
            return self.out(replies + [self.text(K.SERIOUS_SKIN_TEXT)],
                            ["Describe a rash instead", "Start over"], ctx)

        if len(found) >= 2 and re.search(r"\b(differ|difference|vs|versus|compare|or|between)\b", low):
            return self.out(replies + self.compare(found[0], found[1], ctx), self.after_answer_quick(), ctx)

        residual = self.strip_names(low)
        describes = bool(SKIN_RE.search(residual)) or bool(SKIN_RE.search(self.prep(residual)))
        if (found or (aspects and ctx["last"] and not describes)) and (aspects or not describes):
            key = found[0] if found else ctx["last"]
            if key:
                return self.out(replies + self.disease_info(key, aspects, ctx), self.followup_quick(key, aspects), ctx)

        if (set(aspects) - {"what_is"}) and not found and not ctx["last"] and not describes:
            return self.out(replies + [self.text(
                "Happy to explain. Which condition do you mean, or can you describe your symptoms first?")],
                ["Eczema", "Psoriasis", "Acne", "Describe my symptoms"], ctx)

        if has_skin or found:
            r, q = self.add_symptoms(msg, ctx)
            return self.out(replies + r, q, ctx)

        small = self.small_talk(low, words, ctx)
        if small is not None:
            r, q = small
            return self.out(replies + r, q, ctx)

        if replies and replies[0]["type"] == "urgent":
            return self.out(replies, ["Start over"], ctx)

        return self.out(replies + [self.text(
            "I can only help with skin symptoms and common skin conditions. "
            "Tell me what you see or feel on your skin, for example: \"itchy red patches on my elbows\".")],
            self.examples_quick(), ctx)

    def out(self, replies, quick, ctx):
        return {"replies": replies, "quick": quick, "context": ctx}

    # ---------- small talk ----------

    def small_talk(self, low, words, ctx):
        if THANKS_RE.search(low):
            return ([self.text("You're welcome. I'm here if you want to check anything else. "
                               "And please do see a dermatologist if this continues.")], ["Start over"])
        if BYE_RE.search(low):
            return ([self.text("Take care. Remember that I can't diagnose, so see a dermatologist if you are worried.")], [])
        if HOWARE_RE.search(low):
            return ([self.text("Doing well, thanks for asking. How is your skin? Tell me what you've noticed.")],
                    self.examples_quick())
        if IDENT_RE.search(low):
            return ([self.text("I'm a small college project chatbot, not a doctor. I match your description to ten common "
                               "skin conditions and can explain them. I can't diagnose or prescribe.")], self.examples_quick())
        if HELP_RE.search(low):
            return ([self.text("Describe what you see or feel on your skin in your own words. I may ask a few short "
                               "questions to narrow it down, then suggest the closest match from ten common conditions: acne, "
                               "eczema, psoriasis, ringworm, vitiligo, rosacea, contact dermatitis, hives, seborrheic dermatitis "
                               "and scabies. You can also ask things like \"is eczema contagious?\"")], self.examples_quick())
        if GREET_RE.match(low) and len(words) <= 6:
            return ([self.text("Hello. Tell me what you've noticed on your skin: how it looks, where it is and how it feels.")],
                    self.examples_quick())
        if OK_RE.match(low) and len(words) <= 4:
            return ([self.text("Anything else you'd like to add or ask?")],
                    self.after_answer_quick() if ctx["last"] else self.examples_quick())
        if (YES_RE.match(low) or NO_RE.match(low)) and len(words) <= 3:
            return ([self.text("I'm not sure what you're answering. Describe your skin symptoms and I'll take it from there.")],
                    self.examples_quick())
        return None

    # ---------- disease knowledge ----------

    def find_diseases(self, low):
        found = []
        for key, pat in self.alias_patterns:
            if pat.search(low) and key not in found:
                found.append(key)
        return found

    def strip_names(self, low):
        for _, pat in self.alias_patterns:
            low = pat.sub(" ", low)
        return low

    @staticmethod
    def find_aspects(low):
        if "?" not in low and not QUESTION_CUE.search(low):
            return []
        return [name for name, pat in ASPECTS if pat.search(low)][:2]

    def disease_info(self, key, aspects, ctx):
        d = K.DISEASES[key]
        ctx["last"] = key
        if not aspects:
            aspects = ["what_is"]
        replies = []
        for a in aspects:
            if a == "what_is":
                body = f"{d['description']} Typically it appears on: {d['typically_affects'].lower()}."
            elif a == "symptoms":
                body = "Common signs: " + ", ".join(s.lower() for s in d["common_symptoms"]) + "."
            elif a == "prevent":
                body = d["general_advice"]
            else:
                body = d[a]
            replies.append({"type": "info", "title": d["name"], "heading": ASPECT_TITLES[a], "text": body})
        replies.append(self.text("This is general information, not advice about your own skin. "
                                 "If you'd like, describe your symptoms and I'll compare them."))
        return replies

    def followup_quick(self, key, aspects):
        opts = [("contagious", "Is it contagious?"), ("treatment", "How is it treated?"),
                ("causes", "What causes it?"), ("see_doctor", "When to see a doctor?")]
        return [label for a, label in opts if a not in aspects][:3] + ["Describe my symptoms"]

    def compare(self, a, b, ctx):
        da, db = K.DISEASES[a], K.DISEASES[b]
        ctx["last"] = a
        lines = [
            f"{da['name']}: {da['hallmark']}. Usually on: {da['typically_affects'].lower()}.",
            f"{db['name']}: {db['hallmark']}. Usually on: {db['typically_affects'].lower()}.",
        ]
        return [
            {"type": "info", "title": f"{da['name']} vs {db['name']}", "heading": "Main differences", "text": "\n".join(lines)},
            self.text("They can look alike, so a dermatologist is the right person to tell them apart. "
                      "Describe your own symptoms if you want me to compare."),
        ]

    # ---------- symptoms and follow-ups ----------

    def add_symptoms(self, msg, ctx):
        ctx["symptoms"].append(msg)
        ctx["pending"] = None
        if ctx["answered"]:
            ctx["budget"] = len(ctx["asked"]) + 3
            ctx["answered"] = False
        if self.probabilities(ctx) is None:
            ctx["symptoms"].pop()
            return ([self.text("I couldn't match that to skin symptoms I know. Try describing how it looks or feels, "
                               "for example redness, itching, bumps, flaking or color change, and where it is.")],
                    self.examples_quick())
        return self.next_step(ctx, first=len(ctx["symptoms"]) == 1)

    def evidence(self, ctx):
        toks = self.prep(" ".join(ctx["symptoms"])).split()
        return len({t for t in toks if t in self.unigram_set})

    def specific_count(self, ctx):
        toks = self.prep(" ".join(ctx["symptoms"])).split()
        return len({t for t in toks if t in self.specific})

    def next_step(self, ctx, first=False):
        p = self.probabilities(ctx)
        order = np.argsort(p)[::-1]
        top, second = p[order[0]], p[order[1]]
        can_ask = len(ctx["asked"]) < ctx["budget"]
        generic = self.specific_count(ctx) == 0 and not ctx["boosts"]
        thin = (self.evidence(ctx) < 3 and not ctx["boosts"]) or generic
        if can_ask and (thin or top < ASK_BELOW or (top - second) < MIN_MARGIN):
            qid = self.pick_question(ctx, p)
            if qid:
                return self.ask(qid, ctx, first)
        return self.answer(ctx)

    def pick_question(self, ctx, p):
        asked = set(ctx["asked"])
        joined = " ".join(ctx["symptoms"]).lower()
        if "location" not in asked and not BODY_RE.search(joined):
            return "location"
        order = np.argsort(p)[::-1]
        top = [self.classes[i] for i in order[:3] if p[i] >= 0.08]
        if not top:
            return None
        best, best_score = None, 0.0
        for qid, q in K.QUESTIONS.items():
            if qid in asked:
                continue
            hit = [d for d in q["supports"] if d in top]
            if not hit:
                continue
            score = sum(p[self.index[d]] for d in hit)
            if len(top) > 1 and (top[0] in q["supports"]) != (top[1] in q["supports"]):
                score += 0.15
            if score > best_score:
                best, best_score = qid, score
        return best

    def ask(self, qid, ctx, first):
        ctx["pending"] = qid
        n = len(ctx["asked"])
        if qid == "location":
            lead = "Thanks for describing that. A few quick questions will help me narrow it down." if first else "Thanks."
            return ([self.text(lead), {"type": "question", "text": K.LOCATION_QUESTION, "kind": "text"}],
                    K.QUICK_LOCATIONS)
        lead = ["Got it.", "Thanks.", "Okay.", "That helps."][n % 4]
        if n == 0:
            lead = "Thanks for describing that. A few quick questions will help me narrow it down."
        return ([self.text(lead), {"type": "question", "text": K.QUESTIONS[qid]["text"], "kind": "yn"}],
                ["Yes", "No", "Not sure"])

    def handle_pending(self, msg, low, ctx):
        pend = ctx["pending"]
        words = low.split()
        if pend == "location":
            if UNSURE_RE.search(low) or (NO_RE.match(low) and len(words) <= 2):
                ctx["asked"].append("location")
                ctx["pending"] = None
                return self.next_step(ctx)
            if BODY_RE.search(low) or len(words) <= 3:
                ctx["asked"].append("location")
                ctx["pending"] = None
                ctx["symptoms"].append("on my " + msg if len(words) <= 3 else msg)
                if self.probabilities(ctx) is None:
                    ctx["symptoms"].pop()
                return self.next_step(ctx)
            return None
        if len(words) > 5:
            return None
        if UNSURE_RE.search(low):
            verdict = "unsure"
        elif YES_RE.match(low):
            verdict = "yes"
        elif NO_RE.match(low):
            verdict = "no"
        else:
            return None
        supports = K.QUESTIONS[pend]["supports"]
        for d in supports:
            if verdict == "yes":
                ctx["boosts"][d] = ctx["boosts"].get(d, 1.0) * YES_BOOST
            elif verdict == "no":
                ctx["boosts"][d] = ctx["boosts"].get(d, 1.0) * NO_PENALTY
        ctx["asked"].append(pend)
        ctx["pending"] = None
        return self.next_step(ctx)

    def answer(self, ctx, intro=None):
        p = self.probabilities(ctx)
        order = np.argsort(p)[::-1]
        key = self.classes[order[0]]
        top = float(p[order[0]])
        if not ctx["boosts"]:
            if self.evidence(ctx) < 3 or self.specific_count(ctx) == 0:
                top = min(top, 0.35)
            elif self.specific_count(ctx) < 2:
                top = min(top, 0.6)
        d = K.DISEASES[key]
        ctx["last"] = key
        ctx["pending"] = None
        ctx["answered"] = True

        if top >= 0.65:
            label, note = "Strong match", "Your description fits this well."
        elif top >= 0.40:
            label, note = "Moderate match", "This fits best, but other conditions are possible."
        else:
            label, note = "Weak match", "Several conditions fit this description, so treat this as one possibility only."

        alts = []
        for i in order[1:3]:
            if p[i] >= 0.15:
                alts.append({"name": K.DISEASES[self.classes[i]]["name"],
                             "hint": K.DISEASES[self.classes[i]]["hallmark"]})

        asked_any = any(a != "location" for a in ctx["asked"])
        if intro is None:
            intro = ("Thanks, that helps. Putting it together, this is the closest match:" if asked_any
                     else "Based on what you described, this is the closest match:")

        result = {
            "type": "result",
            "disease": {k: d[k] for k in ("name", "description", "common_symptoms", "typically_affects", "general_advice")},
            "confidence": {"label": label, "note": note},
            "alternatives": alts,
            "disclaimer": DISCLAIMER,
        }
        follow = "You can add more details and I'll re-check, or ask me about causes, whether it spreads, or how it is usually managed."
        if CHILD_RE.search(" ".join(ctx["symptoms"]).lower()):
            follow = "Since this involves a child, a pediatrician or dermatologist is the best next step. " + follow
        return ([self.text(intro), result, self.text(follow)], self.after_answer_quick())
