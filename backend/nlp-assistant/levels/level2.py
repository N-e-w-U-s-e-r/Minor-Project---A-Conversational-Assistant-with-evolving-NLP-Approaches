import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

from levels.preprocessing import preprocess


BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_FILE = BASE_DIR / "data" / "intents.json"

THRESHOLD = 0.5


def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def prepare_training_data(intents):
    sentences = []
    labels = []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = preprocess(pattern)
            sentences.append(" ".join(tokens))
            labels.append(intent["name"])

    return sentences, labels


def train_model():
    intents = load_intents()

    sentences, labels = prepare_training_data(intents)

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(sentences)

    model = MultinomialNB()

    model.fit(X, labels)

    return vectorizer, model, X, labels, intents


def predict_intent(text, vectorizer, model, training_vectors):
    tokens = preprocess(text)
    text = " ".join(tokens)

    X = vectorizer.transform([text])

    probabilities = model.predict_proba(X)[0]

    best_index = probabilities.argmax()

    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    similarities = cosine_similarity(
        X,
        training_vectors
    )[0]

    similarity = similarities.max()

    combined_score = (
        0.5 * confidence
        + 0.5 * similarity
    )

    if combined_score < THRESHOLD:
        intent = "unknown"

    return intent, confidence, similarity, combined_score


def get_response(intent, intents):
    for item in intents["intents"]:
        if item["name"] == intent:
            return item["responses"][0]

    return "I don't understand that!"


if __name__ == "__main__":
    vectorizer, model, training_vectors, labels, intents = train_model()

    while True:
        text = input("You: ")

        if text.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        intent, confidence, similarity, combined_score = predict_intent(
            text,
            vectorizer,
            model,
            training_vectors
        )

        print(
            f"Confidence: {confidence:.2f}, "
            f"Similarity: {similarity:.2f}, "
            f"Combined: {combined_score:.2f}"
        )

        print("Assistant:", get_response(intent,intents))
        