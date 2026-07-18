import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load saved objects
vectorizer = joblib.load("vectorizer.pkl")
svd = joblib.load("svd.pkl")
stdscaler = joblib.load("scaler.pkl")
model = joblib.load("stack_extra_xgb.pkl")

# ---------- LABEL MAPPING ----------
label_map = {
    0: "Normal",
    1: "Anxiety",
    2: "Depression",
    3: "Suicidal"
}

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# ---------- SAME PREPROCESS FUNCTIONS ----------
import re, string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text_full(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and word.strip() != ""
    ]

    return " ".join(lemmas)


def extract_text_features(text):
    tokens = text.split()
    if len(tokens) == 0:
        return np.zeros(10)

    return np.array([
        len(tokens),
        len(text),
        np.mean([len(w) for w in tokens]),
        np.std([len(w) for w in tokens]),
        sum(w.isupper() for w in tokens),
        sum(any(c.isdigit() for c in w) for w in tokens),
        sum(w.isalpha() for w in tokens),
        len(set(tokens)),
        len(set(tokens)) / len(tokens),
        sum(len(w) > 6 for w in tokens)
    ])


# ---------- PREDICTION FUNCTION ----------
def predict_text(input_text):
    try:
        clean = preprocess_text_full(input_text)

        # TF-IDF
        X_text = vectorizer.transform([clean])

        # SVD + Scaling
        X_svd = svd.transform(X_text)
        X_svd = stdscaler.transform(X_svd)

        # Statistical features
        stat_features = extract_text_features(clean).reshape(1, -1)

        # Merge
        X_final = np.hstack([X_svd, stat_features])

        # Prediction
        pred_num = model.predict(X_final)[0]
        pred = label_map.get(pred_num, "Unknown")

        return f"Predicted Status: {pred}"

    except Exception as e:
        return f"Error: {str(e)}"


# ---------- GRADIO UI ----------
interface = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs="text",
    title="Mental Health Crisis Detection",
    description="Enter text to predict mental health status"
)

interface.launch()