# nlp/intent.py
# =====================================================================
# LAYER 6 : INTENT DETECTION
# ---------------------------------------------------------------------
# What this file does (in plain English):
#
# The user types a normal English question like
#       "Show me the top 10 customers by revenue"
# and before we can build any SQL we first need to understand WHAT KIND
# of question it is. We call this the "intent".
#
# In this project a question can be one of 7 intents:
#
#       1. KPI            -> asking for a single number
#                            (total / average / count / how much)
#       2. Aggregation    -> group a value by some category
#                            ("sales per city", "revenue by product")
#       3. Ranking        -> top / bottom N of something
#                            ("top 10 customers", "worst 5 stores")
#       4. Comparison     -> compare two or more things
#                            ("Pune vs Mumbai", "difference between ...")
#       5. Trend          -> how a value changes over time
#                            ("monthly sales", "growth over the year")
#       6. Forecasting    -> predict a future value
#                            ("forecast next month", "predict sales")
#       7. Visualization  -> user clearly wants a chart
#                            ("plot", "draw a graph", "show a chart")
#
# HOW WE DETECT THE INTENT (the whole idea in 4 small steps):
#   Step 1  Clean the sentence   -> lowercase, split into words,
#                                   remove common words, lemmatize.
#   Step 2  Turn words -> numbers -> using TF-IDF.
#   Step 3  Ask a Naive Bayes model which intent is most likely.
#   Step 4  If the model is not sure enough, fall back to a very simple
#           keyword search so we never crash.
#
# KEY CONCEPTS (short version):
#   * TF-IDF        : converts words into numbers based on how often a
#                     word appears. Rare-but-meaningful words get a
#                     higher score than common words.
#   * Naive Bayes   : a probability model. It calculates the probability
#                     of each intent given the words, then picks the
#                     intent with the highest probability.
# =====================================================================

import os
import json
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------
# SETTINGS  (all the "magic numbers" and file paths kept in one place)
# ---------------------------------------------------------------------

# Where we save / load the trained model and the training data.
HERE = os.path.dirname(__file__)
MODEL_FILE = os.path.join(HERE, "intent_model.pkl")
TRAIN_FILE = os.path.join(HERE, "train_data.json")

# If the model's best guess is less confident than this, we stop trusting
# it and use the simple keyword fallback instead. 0.35 = 35%.
CONFIDENCE_THRESHOLD = 0.35

# The 7 intents our project supports.
INTENTS = [
    "KPI", "Aggregation", "Ranking", "Comparison",
    "Trend", "Forecasting", "Visualization",
]


# ---------------------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------------------

def setup_nltk():
    """Make sure the NLTK data packs we need are available.

    Wrapped in try/except so the very first run downloads the data
    (stopwords, tokenizer, WordNet) automatically, and every run after
    that is fast (no download).
    """
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

# Create these once (a little expensive to build) and reuse them.
lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words("english"))


# ---------------------------------------------------------------------
# STEP 1 : CLEAN THE TEXT
# ---------------------------------------------------------------------

def clean_text(sentence):
    """Turn a raw question into a clean string of important words.

    Four classic NLP steps:
      1. lowercase        -> "Sales" and "sales" become the same word
      2. tokenize         -> split the sentence into individual words
      3. remove stopwords -> drop common words like "the", "is", "a"
      4. lemmatize        -> reduce a word to its base form
                             ("sales" -> "sale", "running" -> "run")

    Example: "Show me the total sales"  ->  "show total sale"
    """
    words = word_tokenize(sentence.lower())  # 1. lowercase + 2. tokenize

    clean_words = []
    for word in words:
        if not word.isalnum():            # skip punctuation like "?" or ","
            continue
        if word in english_stopwords:     # 3. remove stopwords
            continue
        clean_words.append(lemmatizer.lemmatize(word))  # 4. lemmatize

    # join back into a single string, which is what TF-IDF expects
    return " ".join(clean_words)


# ---------------------------------------------------------------------
# STEP 2 + 3 : BUILD THE MODEL (TF-IDF  ->  NAIVE BAYES)
# ---------------------------------------------------------------------

def build_model():
    """Create the machine learning pipeline.

    A Pipeline chains steps so we treat TF-IDF and Naive Bayes as one model:
      raw text -> [ TF-IDF ] -> numbers -> [ Naive Bayes ] -> intent

    TF-IDF settings:
      ngram_range=(1, 2) -> look at single words AND word pairs, so
                            "next month" is treated as its own signal.
      max_features=500   -> keep only the 500 most useful words/pairs
                            (keeps the model small and avoids noise).
    Naive Bayes settings:
      alpha=0.1          -> Laplace (add-0.1) smoothing, so an unseen word
                            never gets a probability of exactly 0.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=500)),
        ("nb", MultinomialNB(alpha=0.1)),
    ])


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def load_training_data():
    """Read the labelled examples from train_data.json.

    The file is a list of {"query": ..., "intent": ...} objects, e.g.
    {"query": "top 10 products by sales", "intent": "Ranking"}.
    Returns two parallel lists: cleaned questions and their intents.
    """
    with open(TRAIN_FILE) as f:
        data = json.load(f)

    questions = [clean_text(row["query"]) for row in data]
    labels = [row["intent"] for row in data]
    return questions, labels


def train_model():
    """Train the Naive Bayes model, check its accuracy, and save it.

    1. Load and clean the training data.
    2. Measure accuracy with 5-fold cross-validation (train on 4 parts,
       test on 1, repeat 5 times) for an honest accuracy estimate.
    3. Train one final model on ALL the data.
    4. Save the model to disk so we don't have to train again.
    """
    print("[Intent] Loading training data...")
    questions, labels = load_training_data()
    print(f"[Intent] Loaded {len(questions)} labelled queries")

    model = build_model()

    # ---- check accuracy with 5-fold cross-validation ----
    scores = cross_val_score(model, questions, labels, cv=5, scoring="accuracy")
    print(f"[Intent] 5-fold CV accuracy: {scores.mean() * 100:.1f}% "
          f"(+/- {scores.std() * 100:.1f}%)")

    # ---- train the final model on all the data, then save it ----
    model.fit(questions, labels)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"[Intent] Model trained and saved to {MODEL_FILE}")

    return model


def load_model():
    """Load the saved model, or train a new one if none exists yet."""
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    print("[Intent] No saved model found. Training a new one...")
    return train_model()


# Load the model once and keep it here so we don't reload it per question.
_trained_model = None


def get_model():
    """Return the trained model, loading it the first time it is needed."""
    global _trained_model
    if _trained_model is None:
        _trained_model = load_model()
    return _trained_model


# ---------------------------------------------------------------------
# GROUPING-CUE OVERRIDE (KPI vs Aggregation fix)
# ---------------------------------------------------------------------
# The model often labels questions like "Total revenue by region" as KPI
# (a single overall number) when they are really Aggregation (that number
# broken down by a group). The giveaway is a grouping word such as "by",
# "per", "each". A true KPI never has a group breakdown. So when the model
# says KPI but we see a grouping word, we correct it to Aggregation. This
# mirrors the SQL layer (which also promotes these to a GROUP BY query),
# so intent and SQL agree with each other.

print("[Intent] BUILD 2026-07-24-FIXES: grouping-cue KPI->Aggregation override active")

GROUPING_CUE_WORDS = {
    "by", "per", "each", "across", "grouped",
    "breakdown", "segment", "segmented", "distribution",
}


def has_grouping_cue(question):
    """True if the question asks to break a value down by some group."""
    words = set(word_tokenize(question.lower()))
    return len(words & GROUPING_CUE_WORDS) > 0


def correct_kpi_vs_aggregation(question, intent):
    """Fix the common KPI vs Aggregation mix-up.

    A KPI is a single overall number. The moment a question groups that
    number "by region" / "per product" it becomes an Aggregation. So if the
    model guessed KPI but we see a grouping word, switch to Aggregation.
    """
    if intent == "KPI" and has_grouping_cue(question):
        print("[Intent] Correcting KPI -> Aggregation (grouping cue found)")
        return "Aggregation"
    return intent


# ---------------------------------------------------------------------
# STEP 4 (MAIN JOB) : PREDICT THE INTENT
# ---------------------------------------------------------------------

def _predict(question):
    """Shared prediction helper: clean the question, then let Naive Bayes
    predict the intent and return (predicted_intent, top_probability).
    """
    model = get_model()
    clean_question = clean_text(question)
    predicted_intent = model.predict([clean_question])[0]
    # predict_proba gives the probability of every intent; the biggest
    # one is how confident the model is about its answer.
    confidence = max(model.predict_proba([clean_question])[0])
    return predicted_intent, confidence


def detect_intent(question):
    """Return the best intent label for a user's question (as a string).

    This is the main function the rest of the project calls.
      1. clean the question
      2. let Naive Bayes predict the intent and its probabilities
      3. take the highest probability as the "confidence"
      4. if confidence is too low, use the keyword fallback instead
    """
    predicted_intent, confidence = _predict(question)

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[Intent] Low confidence ({confidence:.2f}). "
              f"Using keyword fallback.")
        return correct_kpi_vs_aggregation(question, keyword_fallback(question))

    return correct_kpi_vs_aggregation(question, predicted_intent)


def get_intent_confidence(question):
    """Same as detect_intent, but also returns the confidence as a %.

    Returns (intent, confidence_percent). Handy for the evaluation /
    testing module which wants to print how sure the model was.
    """
    predicted_intent, confidence = _predict(question)
    confidence_percent = round(confidence * 100, 1)

    if confidence_percent < CONFIDENCE_THRESHOLD * 100:
        return (correct_kpi_vs_aggregation(question, keyword_fallback(question)),
                confidence_percent)

    return correct_kpi_vs_aggregation(question, predicted_intent), confidence_percent


# ---------------------------------------------------------------------
# BACKUP PLAN : SIMPLE KEYWORD FALLBACK
# ---------------------------------------------------------------------
# ML is great, but on a tiny training set it can be unsure. When that
# happens we use this simple, easy-to-read keyword matcher so the system
# always returns *some* sensible answer.

# For each intent we list words that strongly hint at it.
INTENT_KEYWORDS = {
    "Forecasting":   ["forecast", "predict", "next", "future", "projection"],
    "Trend":         ["trend", "over time", "monthly", "yearly", "growth",
                      "by month", "by year"],
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

# If two intents tie, we prefer the one higher in this list (more
# specific intents come first).
INTENT_PRIORITY = [
    "Forecasting", "Trend", "Ranking", "Comparison",
    "Visualization", "KPI", "Aggregation",
]


def keyword_fallback(question):
    """Guess the intent by simply counting matching keywords.

    For every intent we count how many of its keywords appear in the
    question. The intent with the most matches wins. If nothing matches at
    all, we default to "Aggregation" (the most common everyday query).
    """
    text = question.lower()
    words = set(word_tokenize(text))

    # count keyword hits for each intent
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = 0
        for keyword in keywords:
            if " " in keyword:          # multi-word phrase -> check whole sentence
                if keyword in text:
                    hits += 1
            elif keyword in words:      # single word -> check the word list
                hits += 1
        scores[intent] = hits

    # pick the best intent, breaking ties using INTENT_PRIORITY
    best_intent = max(
        INTENT_PRIORITY,
        key=lambda intent: (scores[intent], -INTENT_PRIORITY.index(intent)),
    )

    return best_intent if scores[best_intent] > 0 else "Aggregation"


# ---------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------
# Run this file directly (python intent.py) to train the model and see
# how it does on a few example questions.

if __name__ == "__main__":
    print("Training intent classifier...\n")
    train_model()

    print("\nTesting a few example questions:")
    test_questions = [
        "Show top 10 customers by revenue",
        "What is the total profit",
        "Forecast next month sales",
        "Revenue trend by month",
        "Compare sales between Pune and Mumbai",
        "How much did we sell",
        "Which city performed best",
        "Sales numbers month over month",
    ]
    for question in test_questions:
        intent, confidence = get_intent_confidence(question)
        print(f"  {question!r:50} -> {intent} ({confidence}%)")
