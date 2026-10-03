"""
Skin Disease Chatbot - Flask Web Application
=============================================
A simple web interface where users type skin symptoms
and receive an informational prediction.

DISCLAIMER: For academic/educational purposes only.
Not a medical diagnosis tool.
"""

import os
import re
import pickle
from flask import Flask, render_template, request, jsonify

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

MODEL_DIR = "models"

with open(os.path.join(MODEL_DIR, "skin_disease_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_names.pkl"), "rb") as f:
    label_names = pickle.load(f)

DISEASE_INFO = {
    "acne": {
        "name": "Acne",
        "description": "Acne is a common skin condition where hair follicles become clogged with oil and dead skin cells, causing pimples, blackheads, or whiteheads.",
        "common_symptoms": ["Pimples", "Blackheads", "Whiteheads", "Oily skin", "Cysts or nodules"],
        "typically_affects": "Face, forehead, chest, upper back, shoulders",
        "general_advice": "Keep skin clean, avoid touching your face, use non-comedogenic products.",
    },
    "eczema": {
        "name": "Eczema (Atopic Dermatitis)",
        "description": "Eczema causes dry, itchy, and inflamed skin. It is common in children but can occur at any age.",
        "common_symptoms": ["Dry skin", "Itching", "Red patches", "Cracked skin", "Oozing or crusting"],
        "typically_affects": "Elbow creases, behind knees, hands, face",
        "general_advice": "Moisturize regularly, avoid triggers, use gentle skin products.",
    },
    "psoriasis": {
        "name": "Psoriasis",
        "description": "Psoriasis causes red, scaly patches on the skin. It is an autoimmune condition that speeds up skin cell growth.",
        "common_symptoms": ["Red patches", "Silvery scales", "Dry cracked skin", "Itching", "Thickened nails"],
        "typically_affects": "Elbows, knees, scalp, lower back",
        "general_advice": "Moisturize, avoid skin injuries, manage stress.",
    },
    "ringworm": {
        "name": "Ringworm (Tinea)",
        "description": "Ringworm is a fungal infection causing a circular, ring-shaped rash. Despite its name, it is not caused by a worm.",
        "common_symptoms": ["Ring-shaped rash", "Raised scaly edges", "Clear center", "Itching", "Expanding circle"],
        "typically_affects": "Arms, legs, scalp, groin, feet",
        "general_advice": "Keep affected area clean and dry, avoid sharing personal items.",
    },
    "vitiligo": {
        "name": "Vitiligo",
        "description": "Vitiligo causes loss of skin pigment in patches. The affected areas appear white or lighter than the surrounding skin.",
        "common_symptoms": ["White patches", "Loss of skin color", "Premature graying of hair", "Depigmented areas"],
        "typically_affects": "Hands, face, arms, feet, around body openings",
        "general_advice": "Use sun protection on depigmented areas, consult a dermatologist about treatment options.",
    },
    "rosacea": {
        "name": "Rosacea",
        "description": "Rosacea causes redness, visible blood vessels, and sometimes small bumps on the face. It commonly affects the nose and cheeks.",
        "common_symptoms": ["Facial redness", "Visible blood vessels", "Flushing", "Bumps or pimples", "Eye irritation"],
        "typically_affects": "Central face, nose, cheeks, forehead, chin",
        "general_advice": "Avoid triggers (sun, spicy food, alcohol, hot drinks), use gentle skincare.",
    },
    "contact_dermatitis": {
        "name": "Contact Dermatitis",
        "description": "Contact Dermatitis is a rash caused by direct contact with an irritant or allergen such as metals, soaps, or plants.",
        "common_symptoms": ["Red rash", "Itching", "Blisters", "Burning sensation", "Dry cracked skin"],
        "typically_affects": "Hands, face, area of contact with irritant",
        "general_advice": "Identify and avoid the irritant, use protective gloves when needed.",
    },
    "hives": {
        "name": "Hives (Urticaria)",
        "description": "Hives are raised, itchy welts that appear suddenly, often due to an allergic reaction. They usually fade within hours.",
        "common_symptoms": ["Raised welts", "Itching", "Blanching (turns white when pressed)", "Swelling", "Comes and goes"],
        "typically_affects": "Can appear anywhere on the body",
        "general_advice": "Identify triggers, avoid known allergens, seek immediate care if throat swells.",
    },
    "seborrheic_dermatitis": {
        "name": "Seborrheic Dermatitis",
        "description": "Seborrheic Dermatitis causes scaly, flaky, itchy patches on oily areas. Dandruff is a mild form of this condition.",
        "common_symptoms": ["Flaky scalp", "Yellowish greasy scales", "Redness", "Itching", "Dandruff"],
        "typically_affects": "Scalp, eyebrows, nose folds, behind ears, chest",
        "general_advice": "Use medicated shampoos, keep affected areas clean, manage stress.",
    },
    "scabies": {
        "name": "Scabies",
        "description": "Scabies is caused by tiny mites that burrow into the skin, causing intense itching especially at night. It spreads through close contact.",
        "common_symptoms": ["Intense nighttime itching", "Burrow lines", "Red bumps", "Rash between fingers", "Spreads to close contacts"],
        "typically_affects": "Between fingers, wrists, elbows, waistline, groin",
        "general_advice": "See a doctor for prescription treatment; all close contacts should be treated simultaneously.",
    },
}

KEEP_WORDS = {
    "not", "no", "nor", "never", "without",
    "very", "more", "most", "much",
    "few", "all", "both", "each", "every",
    "after", "before", "during", "between",
    "under", "over", "above", "below",
}

stop_words = set(stopwords.words("english")) - KEEP_WORDS
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def predict(symptom_text):
    processed = preprocess_text(symptom_text)
    if len(processed.split()) < 2:
        return None, None
    prediction = model.predict([processed])[0]
    disease_key = label_names[prediction]
    disease_data = DISEASE_INFO.get(disease_key, {})
    return disease_key, disease_data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    symptom_text = data.get("symptoms", "").strip()

    if not symptom_text or len(symptom_text) < 5:
        return jsonify({"error": "Please describe your symptoms in more detail."}), 400

    disease_key, disease_data = predict(symptom_text)

    if disease_key is None:
        return jsonify({"error": "Could not process your input. Please provide a longer description."}), 400

    return jsonify({
        "disease_key": disease_key,
        "disease": disease_data,
        "input": symptom_text,
        "disclaimer": "This is for informational purposes only. Please consult a dermatologist for proper diagnosis.",
    })


if __name__ == "__main__":
    print("Starting Skin Disease Chatbot...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
