# Skin Disease Chatbot using NLP and Machine Learning

A conversational web chatbot that reads a skin problem described in plain English, asks short follow-up questions when it is unsure, and suggests the closest match among 10 common skin conditions. It can also answer general questions such as "is eczema contagious?" or "what is the difference between eczema and psoriasis?".

> **This is an academic project. It is not a medical device and does not diagnose anything.** Every answer says so, and the bot tells users to see a dermatologist.

---

## Table of contents

1. [One-minute summary](#1-one-minute-summary)
2. [Problem statement and goals](#2-problem-statement-and-goals)
3. [What the chatbot can do](#3-what-the-chatbot-can-do)
4. [System architecture](#4-system-architecture)
5. [How to run it](#5-how-to-run-it)
6. [Project structure](#6-project-structure)
7. [The dataset](#7-the-dataset)
8. [Data preprocessing](#8-data-preprocessing)
9. [Feature extraction: TF-IDF](#9-feature-extraction-tf-idf)
10. [Model training and results](#10-model-training-and-results)
11. [The conversation engine](#11-the-conversation-engine)
12. [Follow-up questions: how they work and what they achieve](#12-follow-up-questions-how-they-work-and-what-they-achieve)
13. [Safety and ethics](#13-safety-and-ethics)
14. [The web interface](#14-the-web-interface)
15. [Honest limitations](#15-honest-limitations)
16. [Future work](#16-future-work)
17. [Questions professors may ask, with answers](#17-questions-professors-may-ask-with-answers)
18. [Glossary](#18-glossary)
19. [Tech stack](#19-tech-stack)

---

## 1. One-minute summary

| Item | Value |
|---|---|
| Task | Text classification into 10 skin conditions, wrapped in a dialogue system |
| Classes | acne, eczema, psoriasis, ringworm, vitiligo, rosacea, contact dermatitis, hives, seborrheic dermatitis, scabies |
| Dataset | 2,486 symptom messages, about 250 per class (501 hand-written seed + 1,985 realistic generated) |
| Text representation | TF-IDF, unigrams and bigrams, up to 8,000 features |
| Final model | Logistic Regression (C = 10), chosen because it outputs probabilities |
| First-answer accuracy | **82.1%** on a held-out test set of 498 messages (5-fold CV: 81.5%) |
| Accuracy after follow-up questions | **96.8%** in simulation (about 1.1 questions per chat) |
| Interface | Flask web app with a chat-style UI |
| Everything runs | Locally, on a laptop, in seconds. No GPU, no external API |

**The key idea:** a single classifier answer is right about 8 times in 10, but the right disease is almost always among its top 3 guesses (97%). Instead of guessing, the chatbot looks at how confident the model is, and when it is not confident it asks a short, targeted question that separates the top candidates. The answer is then folded back into the prediction.

---

## 2. Problem statement and goals

**Problem.** People often describe skin problems in everyday words ("itchy red patches on my elbows"). A system that understands such text and gives general information could help people decide whether to seek care. Many online symptom checkers either force fixed menus or give one overconfident guess.

**Goals of this project**

1. Build an NLP pipeline that maps free text to one of 10 common skin conditions, entirely locally with classical machine learning (scikit-learn).
2. Make it behave like a real chatbot: understand greetings, general questions, typos and off-topic input, and ask follow-up questions to improve accuracy.
3. Be responsible: clear disclaimers, urgent-symptom warnings, honest confidence labels.
4. Keep the system simple enough to explain, reproduce and evaluate honestly.

**Non-goals.** Diagnosis, treatment prescriptions, image analysis, or replacing a doctor.

---

## 3. What the chatbot can do

| Capability | Example input | What happens |
|---|---|---|
| Describe symptoms | "my skin is itchy and red" | Asks where it is, then 1 to 3 yes/no questions, then gives a result card |
| Direct clear description | "circular rash on my arm that keeps growing" | Asks at most one check question, then answers |
| Ask about a condition | "is eczema contagious?" | Gives a short factual answer |
| Compare conditions | "difference between eczema and psoriasis" | Shows the key features of each |
| Follow-up about the last result | "how is it treated?" | Answers about the condition it just suggested |
| Small talk | "hi", "thanks", "who are you?" | Short, polite replies |
| Off-topic | "what's the weather?", "my head hurts" | Explains it only handles skin topics |
| Typos and lay words | "small pimpls on my face", "zits" | Corrected or mapped to known words |
| Negation | "itchy but not red and no bumps" | Negated symptoms are ignored, not counted as present |
| Urgent warning | "can't breathe and my lips are swollen" | Shows a red-flag warning telling the user to get emergency care |
| Moles and cancer | "is this mole cancer?" | Says it cannot assess this and advises seeing a dermatologist |
| Skip the questions | "just tell me" | Answers immediately with what it has |

Each result shows the condition name, a plain description, typical symptoms, where it usually appears, general advice, a **confidence label** (Strong, Moderate or Weak match), up to two "also possible" alternatives, and a disclaimer.

---

## 4. System architecture

```
                         +----------------------------+
   User types message -> |  Browser (HTML, CSS, JS)   |
                         |  chat bubbles, quick       |
                         |  replies, keeps context    |
                         +-------------+--------------+
                                       | POST /chat  {message, context}
                                       v
                         +----------------------------+
                         |  Flask app (app.py)        |
                         +-------------+--------------+
                                       |
                                       v
        +--------------------------------------------------------------+
        |  Conversation engine (src/chat_engine.py)                    |
        |                                                              |
        |  1. Safety check (red-flag patterns)                         |
        |  2. Answer a pending follow-up question (yes / no / text)    |
        |  3. Intent routing: small talk, disease question,            |
        |     comparison, off-topic, or symptom description            |
        |  4. Symptom text -> clean -> spell fix -> lemmatize          |
        |  5. TF-IDF + Logistic Regression -> probabilities            |
        |  6. Adjust with yes/no answers                               |
        |  7. Confident enough? answer : ask the best next question    |
        +-------------+------------------------------+-----------------+
                      |                              |
                      v                              v
        +-------------------------+   +----------------------------------+
        | models/*.pkl            |   | src/knowledge.py                 |
        | trained pipeline,       |   | disease facts, 24 follow-up      |
        | label names, word       |   | questions, synonyms, red flags   |
        | purity table            |   +----------------------------------+
        +-------------------------+
```

**Design choice: the server is stateless.** The browser sends the whole conversation state (called `context`) with every message and the server returns the updated state. The context holds the symptoms collected so far, the yes/no adjustments, which questions were already asked and the last suggested condition. This keeps the server simple, makes it easy to test, and means two users never interfere with each other. The server validates and cleans the context on every request, because it comes from the client and must not be trusted.

---

## 5. How to run it

Requirements: Python 3.10 or newer (developed on 3.14).

```bash
pip install -r requirements.txt
```

Run everything from the project root folder.

**Run the chatbot (the model is already trained and included):**

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser.

**Rebuild everything from scratch (full pipeline):**

```bash
python src/build_dataset.py        # merge seed + generated data, remove duplicates
python src/data_preprocessing.py   # clean text, encode labels, train/test split
python src/model_training.py       # train 4 models, compare, save the best
python app.py                      # start the web app
```

**Evaluation and demo scripts:**

```bash
python tests/simulate.py     # simulate users answering follow-up questions on the test set
python tests/try_chat.py     # prints a few sample conversations in the terminal
python tests/try2.py         # more edge cases
```

**Notebook:** `notebooks/skin_disease_chatbot.ipynb` walks through the whole ML part step by step with charts, which is handy for presenting. Open it with Jupyter and run the cells from the top.

> The saved model file depends on the scikit-learn version it was trained with. If you see a version warning on another machine, simply re-run `python src/model_training.py`.

---

## 6. Project structure

```
nlp/
|-- app.py                         Flask server: serves the page, exposes POST /chat
|-- requirements.txt               Python dependencies
|-- README.md                      This file
|
|-- src/
|   |-- build_dataset.py           Merges seed + generated data, removes duplicates
|   |-- data_preprocessing.py      Cleaning, tokenizing, lemmatizing, label encoding, split
|   |-- model_training.py          Trains and compares 4 classifiers, saves the best
|   |-- chat_engine.py             The dialogue engine (intents, follow-ups, prediction)
|   `-- knowledge.py               Disease facts, question bank, synonyms, red-flag patterns
|
|-- data/
|   |-- raw/
|   |   |-- skin_disease_symptoms.csv      501 hand-written seed messages
|   |   |-- generated/                     1,985 realistic messages, one CSV per disease
|   |   `-- skin_disease_dataset.csv       Merged, de-duplicated dataset (2,486 rows)
|   `-- processed/
|       |-- dataset_processed.csv          Cleaned text + label for every row
|       |-- train.csv / test.csv           80% / 20% stratified split
|       `-- label_mapping.csv              Disease name <-> integer
|
|-- models/
|   |-- skin_disease_model.pkl     Trained TF-IDF + Logistic Regression pipeline
|   |-- label_names.pkl            Integer -> disease name
|   `-- word_purity.pkl            For each word, how condition-specific it is
|
|-- templates/index.html           Chat page (markup + JavaScript)
|-- static/style.css               Styling (light and dark mode)
|-- static/fonts/                  Geist fonts, self-hosted
|-- notebooks/                     Step-by-step Jupyter notebook
`-- tests/                         Simulation and demo scripts
```

---

## 7. The dataset

### 7.1 Where could data come from? (data collection)

| Source type | Example | Usable for this project? |
|---|---|---|
| **UCI Dermatology dataset** | 366 patients, 34 numeric clinical and lab features (erythema score 0-3, scaling 0-3, ...) for 6 diseases | **No.** It is *structured* data (numbers from a clinical exam), not text. A user cannot type "erythema = 2". It also covers only 6 diseases and some are not in our list. |
| Image datasets (Kaggle: HAM10000, ISIC, DermNet) | Photos of lesions | Not for a text chatbot. Useful for a future image model. |
| Kaggle and Hugging Face medical text | Symptom-to-disease lists, doctor-patient dialogue sets | Mostly general medicine, rarely skin-specific, rarely with the 10 classes we want. Short keyword lists rather than natural sentences. |
| Medical education sites (DermNet, NHS, Mayo Clinic) | Descriptions of how each condition looks and feels | Good as a *reference* for writing accurate examples. Copying text from them would be a copyright problem. |

**Structured data vs NLP text data.** Structured medical data has fixed numeric columns filled by clinicians. NLP text data is free language written by people, full of slang, spelling errors, vague words and extra context. A chatbot needs the second kind.

**Conclusion.** No public dataset matched "natural-language message plus one of these 10 skin labels", so the dataset was built from two sources, both written specifically for this project.

### 7.2 What we built

| Part | Rows | How it was made | Style |
|---|---|---|---|
| **Seed set** | 501 | Written by hand, about 50 per disease, based on standard descriptions of each condition | Clean, fairly textbook sentences |
| **Generated set** | 1,985 | Written by 10 AI agents, one per disease, each asked for 200 messages with a defined mix of styles | Realistic, messy, varied |
| **Total after cleaning** | **2,486** | Merged and de-duplicated by `src/build_dataset.py` | About 250 per class |

The generated set was requested with this mix, so the data is not all "easy":

- about 25% short fragments or search-style ("itchy flaky scalp since march")
- about 35% one conversational sentence
- about 25% longer, with duration, trigger, what the person already tried and life context
- about 10% sloppy: lowercase, no punctuation, typos, non-native English
- about 5% phrased as questions
- **about 15% deliberately vague or atypical** (for example just "red itchy patches"), where the text alone is ambiguous but the true label is still known
- varied body locations, ages (baby to elderly) and skin tones, with a limit of about 5% of messages naming the diagnosis directly

Message length: median 13 words, mean 15.6, longest 51.

### 7.3 Columns

| Column | Meaning |
|---|---|
| `text` | The patient message (the model input) |
| `disease` | The label (the model output) |
| `body_location`, `severity` | Optional metadata. Not used by the model, kept for analysis |
| `symptoms` | Optional, only present for the seed set |
| `source` | `synthetic_seed` or `synthetic_llm`, so results can be reported per source |

### 7.4 Quality control

- **Exact duplicates** (after lower-casing and removing punctuation): 15 removed.
- **Near duplicates** (same class, word-set overlap of 0.8 or more): checked, 0 found.
- Labels restricted to the 10 allowed values; rows under 8 characters dropped; em-dashes replaced.
- Class balance: 248 to 250 per class, so accuracy is not distorted by one dominant class.

### 7.5 How many samples are enough?

Rule of thumb for a TF-IDF classifier is at least 50 to 100 per class to work and a few hundred per class to be stable. About 250 per class on 10 classes is a reasonable size for a simple model. More realistic variety matters more than raw count, which is why the generated set was designed around variety.

### 7.6 Honest note on the data

The data is **synthetic**: it was written, not collected from real patients. This is stated openly in the report because it limits how far the accuracy numbers can be trusted (see [Limitations](#15-honest-limitations)). Using real patient data would require ethics approval and consent.

---

## 8. Data preprocessing

Implemented in `src/data_preprocessing.py`. The same cleaning is applied again at chat time inside the engine, so training and live use see identical text.

**Steps, in order**

1. **Load** the CSV with pandas.
2. **Inspect**: shape, missing values, class counts, text length statistics.
3. **Remove duplicates** on (text, disease).
4. **Handle missing values**: drop rows with empty text or label; strip whitespace.
5. **Standardize labels** with a mapping (for example "urticaria" becomes `hives`, "tinea" becomes `ringworm`), lower-case, and drop unknown labels.
6. **Clean text**: lower-case; remove URLs and emails; replace punctuation and special characters with spaces (keeping hyphens); collapse repeated spaces.
7. **Tokenize** with NLTK.
8. **Remove stopwords conservatively** (see below).
9. **Lemmatize** with WordNet ("patches" becomes "patch", "bumps" becomes "bump").
10. **Drop very short messages** (fewer than 2 informative words).
11. **Encode labels** to integers 0 to 9 with a label encoder; the mapping is saved.
12. **Stratified 80/20 split**, `random_state = 42`: 1,988 training rows and 498 test rows with the same class proportions in both.

**Example**

| Raw | Processed | Label |
|---|---|---|
| "I have itchy red patches on my elbows and knees that keep coming back" | `itchy red patch elbow knee keep coming back` | psoriasis |
| "Little red bumps showed up around my waist and belt line and the itch is way worse once I lie down." | `little red bump showed around waist belt line itch way worse lie` | scabies |

### 8.1 What we deliberately do NOT do (and why)

In medical text, small words carry meaning. Standard NLP clean-up can destroy it.

| Step avoided | Why |
|---|---|
| Removing negations ("no", "not", "without", "never") | "itchy" and "not itchy" mean opposite things |
| Removing intensity and timing words ("very", "more", "after", "during", "between") | "after eating" suggests hives; "between my fingers" suggests scabies |
| Removing numbers | Durations and counts can matter ("3 weeks") |
| Removing colour, body-part or texture words | These are the core symptom information |
| Stemming (chopping words crudely) | Can merge unrelated words; lemmatization is gentler |
| Removing rare words | A rare word like "burrow" can be the single best clue |

So the stopword list is the standard English list **minus about 22 protected words**. Negation is handled explicitly later by the engine (section 11).

---

## 9. Feature extraction: TF-IDF

A model needs numbers, not text. We use **TF-IDF** (Term Frequency - Inverse Document Frequency).

- **TF**: how often a word occurs in this message.
- **IDF**: how rare the word is across all messages. A word like "skin" appears everywhere and gets a low weight. A word like "silvery" appears in few messages and gets a high weight.
- A word's score is TF multiplied by IDF, so words that are frequent here but rare overall are treated as most informative.

Settings used:

| Setting | Value | Reason |
|---|---|---|
| n-grams | 1 and 2 | Captures pairs such as "ring shaped", "worse night", "between fingers" |
| `max_features` | 8,000 | Keeps the most useful words and pairs; limits overfitting |
| `sublinear_tf` | True | Uses 1 + log(TF) so repeating a word many times does not dominate |

**Why TF-IDF and not a neural network or BERT?** The goal was a simple model that trains in seconds on a laptop, can be explained line by line, and is easy to inspect. With about 2,500 short messages, TF-IDF with a linear model is a strong baseline, and linear models also give probabilities and interpretable word weights.

---

## 10. Model training and results

Implemented in `src/model_training.py`. Four classifiers were trained on the same features and compared on the same split.

| Model | 5-fold CV accuracy | Test accuracy |
|---|---|---|
| Multinomial Naive Bayes | 77.2% | 78.1% |
| Linear SVM | 81.3% | 81.9% |
| **Logistic Regression (C = 10)** | **81.5%** | **82.1%** |
| Random Forest | 75.4% | 79.1% |

(Cross-validation is run on the training portion only; the test set is touched once per model.)

### 10.1 Why Logistic Regression was chosen

Linear SVM is almost the same in accuracy, but it does not produce probabilities. The chatbot's follow-up logic needs a probability for every disease so it can decide *whether it is confident* and *which diseases to ask about*. So only models that can output probabilities are eligible, and among those Logistic Regression is best. This rule is written into the training script.

### 10.2 Choosing the regularization strength

| C | CV | Test | Average top probability |
|---|---|---|---|
| 5 | 81.4% | 82.5% | 0.63 |
| **10** | **81.5%** | **82.1%** | **0.69** |
| 30 | 81.2% | 81.9% | 0.76 |
| 100 | 80.8% | 81.7% | 0.81 |

C = 10 had the best cross-validation score, and its average confidence (0.69) is close to its real accuracy (about 82%). Higher C values become overconfident: they report 0.76 to 0.81 certainty while being right only about 82% of the time. Well-calibrated confidence matters here because the chatbot uses it to decide when to ask questions.

### 10.3 Per-class results (Logistic Regression, test set)

| Disease | Precision | Recall | F1 |
|---|---|---|---|
| acne | 0.94 | 0.90 | 0.92 |
| vitiligo | 0.92 | 0.96 | 0.94 |
| rosacea | 0.81 | 1.00 | 0.89 |
| scabies | 0.82 | 0.92 | 0.87 |
| seborrheic dermatitis | 0.86 | 0.84 | 0.85 |
| hives | 0.79 | 0.84 | 0.81 |
| ringworm | 0.85 | 0.78 | 0.81 |
| psoriasis | 0.69 | 0.74 | 0.71 |
| contact dermatitis | 0.81 | 0.60 | 0.69 |
| eczema | 0.74 | 0.63 | 0.68 |

**Reading this.** Conditions with very distinctive wording (vitiligo: white patches, acne: pimples and blackheads) are easy. The weak classes are **eczema, contact dermatitis and psoriasis**. That is realistic: in real life these three look similar (red, itchy, dry, scaly patches) and even clinicians confuse them without examination. Most errors are between these three.

Other test results:

- The correct disease is within the model's **top 2** guesses about 94% of the time and within the **top 3** about 97% of the time. This is why follow-up questions work so well.
- Accuracy by data source: about 81% on the new realistic messages and about 86% on the older clean seed messages. The realistic messages are harder, as intended.

### 10.4 Which words drive the predictions

Because the model is linear, each word has a weight per disease, so the decisions can be inspected. Examples of strongly weighted words: "pimple", "blackhead" (acne); "ring", "circular" (ringworm); "welt", "come and go" (hives); "night", "burrow", "finger" (scabies); "white", "depigmented" (vitiligo); "flush", "vein" (rosacea).

---

## 11. The conversation engine

Implemented in `src/chat_engine.py` (logic) and `src/knowledge.py` (facts and word lists). This is what turns a classifier into a chatbot.

### 11.1 Processing order for every message

1. **Red-flag check.** If the message matches an urgent pattern, an urgent warning is added to the reply (details in section 13).
2. **Reset words** ("start over", "new chat") clear the conversation.
3. **Pending question.** If the bot just asked something, try to read the message as an answer:
   - yes / no / not sure (accepts "yeah", "nope", "kind of", "idk", ...) for yes/no questions;
   - a location or free text for the "where is it?" question;
   - if the message is clearly something else, it is handled normally instead.
4. **"Just tell me"** answers immediately with the information collected so far.
5. **Mole or cancer words** get a safe redirect to a dermatologist.
6. **Two diseases plus "difference/vs/or"** gives a comparison.
7. **Questions about a condition** ("is it contagious?", "what causes eczema?") are answered from the knowledge base. If no condition is named, it uses the last one suggested.
8. **Symptom description**: the text is added to the conversation and the prediction logic runs (section 12).
9. **Small talk** (greetings, thanks, who are you, help).
10. **Anything else** gets a polite "I only help with skin topics" with example prompts.

**Telling a question from a symptom description.** "Is eczema contagious?" and "my eczema bleeds when I scratch it" both mention eczema. The engine removes the disease names from the text and checks whether what remains describes skin symptoms. If yes, it is treated as a symptom description; if not, as a knowledge question. Condition aspects such as "contagious" or "treatment" are only recognised when the message is actually phrased as a question, so "I put cream on it but it's still itchy" is not mistaken for a treatment question.

### 11.2 Understanding messy input

| Problem | How it is handled |
|---|---|
| **Spelling mistakes** ("pimpls") | Words not in the model vocabulary are matched to the closest vocabulary word (difflib similarity of 0.8 or more, same first letter) |
| **Lay words** ("zits", "scratchy", "my kid") | A small synonym table maps them to words seen in training; words already in training are deliberately not remapped, so no signal is lost |
| **Negation** ("not red", "no bumps", "doesn't itch") | Contractions are expanded, then the negation word plus the word after it is removed. The model cannot read "not itchy" correctly as a bag of words, so ignoring negated words is safer than counting them |
| **Contractions and punctuation** | Normalized the same way as in training |
| **Off-topic text** | A curated list of skin-related word stems decides whether a message is about skin at all. Words are also checked after spelling correction, so typos do not look off-topic |

### 11.3 Knowledge base

`src/knowledge.py` stores, for each of the 10 conditions: description, typical symptoms, typical locations, a one-line "hallmark" feature, general advice, causes, whether it is contagious, general management (non-prescriptive), and when to see a doctor. These were written from standard patient-information sources and phrased as general information only.

---

## 12. Follow-up questions: how they work and what they achieve

### 12.1 Why ask questions at all

Because a message like "itchy red patches" really does fit many diseases. A one-shot guess would be wrong often and confidently so. Asking one or two well-chosen questions resolves most of that ambiguity, like a doctor taking a history.

### 12.2 When the bot asks versus answers

After each message the model produces a probability for each disease. The bot **asks** a question if any of these hold (and the question budget is not used up):

- the top probability is below **0.70**, or
- the gap between the top two diseases is below **0.30**, or
- the message contains **no condition-specific words** (see below), or
- there are fewer than 3 informative words.

Otherwise it answers.

The budget is **4 questions** per case (the location question counts as one). If new details arrive after an answer, 3 more questions are allowed.

### 12.3 Condition-specific words (a guard against overconfidence)

A message like "my skin is itchy and red" is nearly meaningless to the model, but a classifier can still output a high probability for the most common class. To avoid this, a **purity table** is computed at training time: for each word, the share of training messages containing it that belong to its most common disease, plus how many messages contain it. A word counts as *specific* when it appears in at least 6 messages and at least 60% of them are one disease ("pimple", "ring", "welt", "night"). Generic words ("itchy", "red", "skin", "rash") fail this test. If a message has no specific word, the bot always asks, and the confidence label is capped at Weak. If it has only one, the label is capped at Moderate.

### 12.4 Choosing the best question

There are 24 yes/no questions, each linked to the diseases a "yes" supports, for example:

| Question | A "yes" supports |
|---|---|
| "Is the itching much worse at night?" | scabies, eczema |
| "Is the rash ring-shaped, with a raised edge and a clearer center?" | ringworm |
| "Are the red patches covered with thick, silvery-white scales?" | psoriasis |
| "Did it start after touching something new (jewelry, soap, a plant...)?" | contact dermatitis |
| "Do the bumps come and go within hours?" | hives |
| "Does your face flush easily with heat, spicy food or alcohol?" | rosacea |

The question to ask next is chosen like this:

1. If no body location has been mentioned yet, ask where it is.
2. Otherwise take the top 3 candidate diseases (probability at least 0.08).
3. For every unasked question, score = sum of the probabilities of the candidate diseases it supports.
4. Add a bonus of 0.15 if the question supports the 1st-ranked candidate but not the 2nd (or the reverse), because that separates the two leaders.
5. Ask the highest-scoring question.

### 12.4b Using the answers

- **Yes**: probabilities of the supported diseases are multiplied by **2.5**.
- **No**: multiplied by **0.35**.
- **Not sure / skip**: no change.
- After each change the probabilities are re-normalized to sum to 1, then the ask-or-answer rule runs again.

These multipliers act like a simple Bayesian update (a likelihood ratio applied to a prior). They are a transparent hand-set heuristic rather than learned values, which is a stated limitation.

### 12.5 The final answer

| Probability of top disease | Label |
|---|---|
| 0.65 or more | Strong match |
| 0.40 to 0.65 | Moderate match |
| below 0.40 | Weak match, with the note that several conditions fit |

Any other disease with probability of at least 0.15 (maximum two) is listed as "Also possible", with a one-line description of how it usually looks, so the user sees that the answer is not certain.

### 12.6 Measured effect (simulation)

`tests/simulate.py` takes each of the 498 held-out test messages, gives it to the chatbot, and answers every follow-up question **truthfully** according to the known correct disease (yes if the question supports it, otherwise no; "skip" for location).

| Measure | Result |
|---|---|
| First-answer accuracy (before any question) | 82.5% |
| Accuracy after follow-up questions | **96.8%** |
| Average questions asked | 1.14 |
| Messages the engine could not parse at all | 3 of 498 |

**Read this result carefully when presenting it.** It is an *upper-bound style* estimate: it assumes users answer correctly and understand the questions, and it feeds the already-cleaned text of the test set. Real users will be less consistent, so real improvement will be smaller. It still shows the mechanism works, because the right answer is nearly always among the top 3 candidates and the questions are designed to separate exactly those.

---

## 13. Safety and ethics

- **No diagnosis claims.** A disclaimer appears in the header, the greeting, the input area and every result.
- **Honest confidence.** The bot shows Strong, Moderate or Weak and lists alternatives, instead of presenting one guess as fact. Vague input cannot produce a Strong match.
- **Urgent-symptom detection.** Patterns for serious situations trigger a prominent warning to seek emergency care: difficulty breathing, swelling of the lips, tongue, throat or face, fever together with rash or blisters, skin peeling off in sheets, blisters in the mouth or eyes, rapidly spreading redness or red streaks, pus with fever or chills, fainting or dizziness. The warning appears even if a prediction is also made.
- **Moles and skin cancer** are explicitly out of scope; the bot redirects to a dermatologist.
- **Children.** If a child is mentioned, the answer adds that a pediatrician or dermatologist is the best next step.
- **Not prescriptive.** Management information is general ("moisturize", "antihistamines are the usual first step") with no doses and no personal treatment plans, and always says when to see a doctor.
- **Privacy.** Nothing is stored on the server and there is no database or login. The conversation lives only in the browser tab and disappears when it is closed. No third-party service is called.
- **Input safety.** Messages are length-limited, all output is HTML-escaped in the browser (prevents script injection), and the client-supplied context is validated and clamped on the server (types, lengths, allowed keys, multiplier ranges).
- **Data ethics.** No real patient data was used.

---

## 14. The web interface

Built with plain HTML, CSS and JavaScript (no framework), served by Flask.

- Chat window with bubbles, avatars, timestamps and a typing indicator.
- **Quick-reply buttons** under questions (Yes / No / Not sure; location choices) and under answers (Is it contagious? How is it treated? When to see a doctor? Start over).
- The input hint changes to match what the bot asked.
- Result card with name, description, symptom tags, location, advice, confidence, alternatives and disclaimer.
- Red-bordered urgent message for red-flag input; inline red error message for server problems.
- Light and dark mode follow the system setting. Responsive layout for phones.
- Accessibility: keyboard focus styles, skip link, live region for new messages, reduced-motion support.
- The Geist font is stored locally in `static/fonts/`, so nothing is loaded from the internet.

**API.** `POST /chat` with JSON `{"message": "...", "context": {...}}`. It returns `{"replies": [...], "quick": [...], "context": {...}}`. Each reply has a type: `text`, `question`, `info`, `result` or `urgent`.

---

## 15. Honest limitations

1. **The data is synthetic.** It was written by people and AI to resemble real messages; it is not real patient text. Real users may phrase things differently, so real accuracy is likely lower than the reported numbers.
2. **Test data comes from the same source as training data.** The split is random within one dataset, so the 82% is not a measure of performance on genuinely new, outside text. The seed messages are also fairly similar to each other, which is why results are reported separately for the more realistic generated messages (81%) and the older seed messages (86%).
3. **Only 10 conditions.** Anything else (skin cancer, shingles, impetigo, drug rashes, and so on) will be forced into one of the 10 or ignored. The bot says it cannot assess moles.
4. **Text only.** Skin conditions are largely visual; text is a weak substitute for a photo and an examination.
5. **Bag-of-words model.** It does not understand word order or deeper meaning; negation is only handled by a simple rule.
6. **The follow-up multipliers (2.5 and 0.35) and thresholds (0.70, 0.30, 0.65, 0.40) are hand-set.** They were chosen by reasoning and checked by simulation, not learned from real conversations.
7. **The simulation assumes ideal answers**, so the 96.8% is optimistic.
8. **Confusable classes.** Eczema, psoriasis and contact dermatitis stay the hardest, and they are hard for people too.
9. **English only.**
10. **Not clinically validated.**

---

## 16. Future work

- Collect real, consented symptom descriptions (with ethics approval) and use them as a separate test set.
- Compare with transformer models (for example fine-tuning a small BERT) and measure whether the extra complexity pays off.
- Learn the follow-up question weights from real conversations instead of hand-set multipliers; use information gain to pick questions.
- Probability calibration (for example Platt scaling or isotonic regression) and a calibration plot.
- Handle more conditions and a proper "none of these" class.
- Add photo input with an image model for a multimodal system.
- Multi-language support.
- Add an automated test suite and continuous evaluation.

---

## 17. Questions professors may ask, with answers

**Q: Why classical ML and not deep learning or an LLM?**
A: The goal was a transparent, locally trainable system. With about 2,500 short texts, TF-IDF plus Logistic Regression reaches 82% and trains in seconds. It is easy to inspect (word weights), gives calibrated probabilities that the dialogue logic needs, and needs no GPU or external service. A transformer would likely help on wording variety, and it is listed as future work.

**Q: Where did the data come from? Is it real?**
A: No public dataset has free-text symptom messages labelled with these 10 skin diseases. The UCI Dermatology dataset is numeric clinical features, not text. So we built the dataset: 501 hand-written seed messages plus 1,985 realistic messages written to a specification (varied length, slang, typos, vague cases, different ages and skin tones). It is synthetic, and we state that as a limitation. Real data would need ethics approval.

**Q: Is 82% accuracy good?**
A: For 10 classes where three of them look alike even to humans, 82% on realistic text is solid for a baseline. Random guessing would be 10%. More importantly the correct answer is in the top 3 about 97% of the time, which is what makes follow-up questions effective. It is not good enough to diagnose anyone, which is why the system never claims to.

**Q: How do follow-up questions improve accuracy?**
A: The model outputs probabilities. When it is unsure, the bot picks a yes/no question that separates the leading candidates. A "yes" multiplies the supported diseases' probabilities by 2.5, a "no" by 0.35, and the probabilities are re-normalized. In simulation this raised accuracy from 82.5% to 96.8% with 1.14 questions on average. This is an optimistic estimate because the simulated user always answers correctly.

**Q: Why not just use the highest-probability answer every time?**
A: Because for vague input the top probability is often low or the gap to the second choice is small, and a wrong confident answer is worse than a question in a health setting. The bot also shows a confidence label and alternatives.

**Q: Why did you choose Logistic Regression over the slightly simpler SVM?**
A: Same accuracy, but Logistic Regression gives probabilities, which the follow-up logic requires. SVMs only give a decision score without calibrated probabilities.

**Q: What is TF-IDF and why bigrams?**
A: TF-IDF weights a word by how often it appears in the message and how rare it is across messages, so common words count less. Bigrams capture short phrases like "ring shaped" or "between fingers" that single words miss.

**Q: How do you prevent overfitting?**
A: Held-out test set never used for training or tuning decisions on the final number; 5-fold cross-validation on the training set; regularization (C tuned on cross-validation); capped vocabulary (8,000 features); duplicate removal. Cross-validation (81.5%) and test accuracy (82.1%) agree, which suggests the model is not just memorizing.

**Q: Is there data leakage?**
A: Exact duplicates were removed. The seed messages are fairly similar to each other, so a random split can place similar sentences in both train and test; that is why we report the generated realistic messages separately. A fully independent real-world test set is the proper next step, and we say so in the limitations.

**Q: Why do you keep words like "not", "no" and "very" when most NLP removes stopwords?**
A: In medical text they change meaning. "not itchy" is different from "itchy". We keep about 22 such words, and the engine additionally drops the word that follows a negation, because a bag-of-words model cannot interpret negation itself.

**Q: How does the bot understand questions like "is eczema contagious?" without a language model?**
A: It uses rule-based intent detection: question cues (how, what, is, can...) plus topic patterns (contagious, treat, causes, doctor...), matched with regular expressions, and answers from a curated knowledge base. It first checks whether the rest of the sentence describes symptoms, so "my eczema bleeds" is treated as a symptom description rather than a knowledge question.

**Q: What stops it from making dangerous statements?**
A: Disclaimers everywhere, no diagnosis wording, no prescriptions or doses, emergency pattern detection with a clear call to seek urgent care, redirection for moles and cancer, honest confidence, and alternatives. It is a limited information tool, not a clinician.

**Q: How does it handle misspellings and slang?**
A: Misspelled words are matched to the closest known vocabulary word, a small synonym table maps everyday words to training words, and off-topic detection also looks at the spell-corrected text.

**Q: Why is the server stateless?**
A: The browser sends the conversation state with every request. This is simple, scalable and private (nothing is stored), and the server validates it each time because it comes from the client.

**Q: What would you do with more time?**
A: Gather real consented data for an independent test set, compare against a fine-tuned transformer, learn the question weights from real dialogues, add calibration plots, support more conditions and add image input.

**Q: What was the hardest part?**
A: Making the bot appropriately uncertain. A classifier tends to be confident on vague input. We needed purity-based "specific word" detection, confidence caps and calibrated regularization so that "my skin is itchy and red" triggers questions instead of a confident guess.

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **NLP** | Natural Language Processing: making computers work with human language |
| **Classification** | Predicting which category an input belongs to |
| **Tokenization** | Splitting text into words |
| **Stopword** | A very common word (the, is, on) usually removed in NLP |
| **Lemmatization** | Reducing a word to its dictionary form (patches to patch) |
| **TF-IDF** | A weighting that scores words by frequency in a message and rarity overall |
| **n-gram** | A sequence of n words (bigram = 2 words) |
| **Logistic Regression** | A linear model that outputs a probability for each class |
| **Regularization (C)** | A penalty that keeps the model simple. Larger C means weaker penalty |
| **Cross-validation** | Splitting training data into folds, training on some and checking on the rest, to estimate performance |
| **Stratified split** | A split that keeps the same class proportions in train and test |
| **Precision / Recall / F1** | Precision: of the messages predicted as X, how many were X. Recall: of all real X, how many were found. F1 combines both |
| **Top-k accuracy** | The correct answer is among the k highest-probability guesses |
| **Calibration** | Whether predicted confidence matches real accuracy |
| **Intent** | What the user is trying to do (greet, ask a question, describe symptoms) |
| **Red flag** | A symptom pattern that may need urgent medical care |
| **Stateless server** | A server that keeps no memory between requests |

---

## 19. Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3 |
| Data | pandas, NumPy |
| NLP | NLTK (tokenization, stopwords, WordNet lemmatizer), difflib (spell matching), regular expressions |
| Machine learning | scikit-learn (TfidfVectorizer, LogisticRegression, LinearSVC, MultinomialNB, RandomForest, cross-validation, metrics) |
| Web | Flask, HTML, CSS, JavaScript (no framework) |
| Notebook | Jupyter, matplotlib, seaborn |
| Fonts | Geist and Geist Mono (self-hosted) |

---

*Academic project. For information only. Not medical advice. If you have a skin concern, see a dermatologist.*
