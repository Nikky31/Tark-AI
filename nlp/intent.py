import os
import json
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score

HERE = os.path.dirname(__file__)
MODEL_FILE = os.path.join(HERE, "intent_model.pkl")
TRAIN_FILE = os.path.join(HERE, "train_data_augmented.json")

CONFIDENCE_THRESHOLD = 0.35

INTENTS = [
    "KPI", "Aggregation", "Ranking", "Comparison",
    "Trend", "Visualization",
]

def setup_nltk():
    try:
        stopwords.words("english")
        word_tokenize("test sentence")
    except LookupError:
        print("[Intent] Downloading NLTK data (only happens once)...")
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)

setup_nltk()

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))

def clean_text(sentence):
    words = word_tokenize(sentence.lower())  

    clean_words = []
    for word in words:
        if not word.isalnum():            
            continue
        if word in english_stopwords:    
            continue
        clean_words.append(lemmatizer.lemmatize(word))  

    return " ".join(clean_words)

#random forest
def to_dense(matrix):
    return matrix.toarray()


def build_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                 sublinear_tf=True)),
    ])

#ml pipepline
def build_model():
    return Pipeline([
        ("features", build_features()),
        ("to_dense", FunctionTransformer(to_dense, accept_sparse=True)),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="log2",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

#train model 12/6/26
def load_training_data():
    with open(TRAIN_FILE) as f:
        data = json.load(f)

    questions = [clean_text(row["query"]) for row in data]
    labels = [row["intent"] for row in data]
    return questions, labels


def train_model():
    print("[Intent] Loading training data...")
    questions, labels = load_training_data()
    print(f"[Intent] Loaded {len(questions)} labelled queries")

    model = build_model()

    #  accuracy - 5-fold cross-validation 
    scores = cross_val_score(model, questions, labels, cv=5, scoring="accuracy")
    print(f"[Intent] 5-fold CV accuracy: {scores.mean() * 100:.1f}% "
          f"(+/- {scores.std() * 100:.1f}%)")

    
    model.fit(questions, labels)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"[Intent] Model trained and saved to {MODEL_FILE}")

    return model


def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    print("[Intent] No saved model found. Training a new one...")
    return train_model()

_trained_model = None


def get_model():
    """Return the trained model, loading it the first time it is needed."""
    global _trained_model
    if _trained_model is None:
        _trained_model = load_model()
    return _trained_model

print("[Intent] BUILD 2026-07-24-FIXES: grouping-cue KPI->Aggregation override active")

GROUPING_CUE_WORDS = {
    "by", "per", "each", "across", "grouped",
    "breakdown", "segment", "segmented", "distribution",
}


def has_grouping_cue(question):
    words = set(word_tokenize(question.lower()))
    return len(words & GROUPING_CUE_WORDS) > 0


def correct_kpi_vs_aggregation(question, intent):
    if intent == "KPI" and has_grouping_cue(question):
        print("[Intent] Correcting KPI -> Aggregation (grouping cue found)")
        return "Aggregation"
    return intent

def _predict(question):
    model = get_model()
    clean_question = clean_text(question)
    predicted_intent = model.predict([clean_question])[0]
    confidence = max(model.predict_proba([clean_question])[0])
    return predicted_intent, confidence


def detect_intent(question):
    predicted_intent, confidence = _predict(question)

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[Intent] Low confidence ({confidence:.2f}). "
              f"Using keyword fallback.")
        return correct_kpi_vs_aggregation(question, keyword_fallback(question))

    return correct_kpi_vs_aggregation(question, predicted_intent)


def get_intent_confidence(question):
    predicted_intent, confidence = _predict(question)
    confidence_percent = round(confidence * 100, 1)

    if confidence_percent < CONFIDENCE_THRESHOLD * 100:
        return (correct_kpi_vs_aggregation(question, keyword_fallback(question)),
                confidence_percent)

    return correct_kpi_vs_aggregation(question, predicted_intent), confidence_percent



# backup:-key word fallback 13/6/2026
INTENT_KEYWORDS = {
    "Trend":         ["trend", "over time", "monthly", "yearly", "growth",
                      "by month", "by year", "forecast", "predict", "next",
                      "future", "projection"],
    "Ranking":       ["top", "bottom", "highest", "lowest", "rank", "best",
                      "worst"],
    "Comparison":    ["compare", "comparison", "versus", "vs", "between",
                      "difference"],
    "Visualization": ["chart", "plot", "graph", "visualize", "treemap",
                      "show me"],
    "KPI":           ["kpi", "total", "average", "avg", "count", "sum",
                      "metric", "how much", "how many", "what is the"],
    "Aggregation":   ["group", "by", "per", "each", "breakdown"],
}

INTENT_PRIORITY = [
    "Trend", "Ranking", "Comparison",
    "Visualization", "KPI", "Aggregation",
]


def keyword_fallback(question):
    text = question.lower()
    words = set(word_tokenize(text))
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = 0
        for keyword in keywords:
            if " " in keyword:         
                if keyword in text:
                    hits += 1
            elif keyword in words:    
                hits += 1
        scores[intent] = hits
    best_intent = max(
        INTENT_PRIORITY,
        key=lambda intent: (scores[intent], -INTENT_PRIORITY.index(intent)),
    )

    return best_intent if scores[best_intent] > 0 else "Aggregation"

if __name__ == "__main__":
    print("Training intent classifier...\n")
    train_model()

    #test
    print("\nTesting a few example questions:")
    test_questions = [
        "Show top 10 customers by revenue",
        "What is the total profit",
        "Revenue trend by month",
        "Compare sales between Pune and Mumbai",
        "How much did we sell",
        "Which city performed best",
        "Sales numbers month over month",
    ]
    for question in test_questions:
        intent, confidence = get_intent_confidence(question)
        print(f"  {question!r:50} -> {intent} ({confidence}%)")
