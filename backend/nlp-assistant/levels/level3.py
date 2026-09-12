import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INTENTS_FILE = BASE_DIR / "data" / "intents.json"
VALIDATION_FILE = BASE_DIR / "data" / "validation.json"


from levels.preprocessing import preprocess


THRESHOLD = 0.3

def evaluate_model(model, vectorizer, validation_data):
    known_correct = 0
    known_total = 0

    unknown_correct = 0
    unknown_total = 0

    print("\nEvaluation:")

    for item in validation_data["validation"]:
        text = item["text"]
        expected = item["expected"]

        intent, confidence, probabilities = predict_intent(
            text,
            vectorizer,
            model
        )

        if expected == "unknown":
            unknown_total += 1

            if intent == "unknown":
                unknown_correct += 1

                print(
                    f"{text} → {intent} "
                    f"({confidence:.2f}) "
                    f"[expected: unknown]"
                )

        else:
            known_total += 1

            if intent == expected:
                known_correct += 1

            print(
                f"{text} → {intent} "
                f"({confidence:.2f}) "
                f"[expected: {expected}]"
            )

    print(
        f"\nKnown accuracy: "
        f"{known_correct / known_total:.2%}"
    )

    print(
        f"Unknown examples: "
        f"{unknown_total}"
    )
    print(
        f"\nUnknown accuracy: "
        f"{unknown_correct / unknown_total:.2%}"
    )

def validate(model, vectorizer, validation_data):
    print("\nValidation:")

    for item in validation_data["validation"]:
        text = item["text"]
        expected = item["expected"]

        intent, confidence, probabilities = predict_intent(
            text,
            vectorizer,
            model
        )

        print(
            f"{text} → {intent} "
            f"({confidence:.2f}) "
            f"[expected: {expected}]"
        )

def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def load_validation_data():
    with open(VALIDATION_FILE, "r", encoding="utf-8") as file:
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


def train_model(classifier="logistic"):
    intents = load_intents()

    sentences, labels = prepare_training_data(intents)

    print("\n=== TRAINING DATA ===")

    for sentence, label in zip(sentences, labels):
        print(f"{label:30} -> {sentence}")

    vectorizer = TfidfVectorizer(ngram_range=(1,2))

    X = vectorizer.fit_transform(sentences)

    if classifier=="logistic":
        model = LogisticRegression()
    elif classifier=="svm":
        base_model = LinearSVC(random_state=42)
        model = CalibratedClassifierCV(
            base_model,
            method="sigmoid",
            cv=5
        )
    else:
        raise ValueError("Unknown classifier")

    model.fit(X, labels)

    training_predictions = model.predict(X)

    print("\n=== TRAINING ACCURACY ===")

    correct = 0

    for sentence, actual, predicted in zip(
        sentences,
        labels,
        training_predictions
    ):
        if actual == predicted:
            correct += 1
        else:
            print(
                f"{sentence} → predicted: {predicted}, "
                f"expected: {actual}"
            )

    print(
        f"\nTraining accuracy: "
        f"{correct / len(labels):.2%}"
    )

    return vectorizer, model, intents


def predict_intent(text, vectorizer, model):
    tokens = preprocess(text)
    text = " ".join(tokens)

    X = vectorizer.transform([text])

    probabilities = model.predict_proba(X)[0]

    best_index = probabilities.argmax()

    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < THRESHOLD:
        intent = "unknown"

    return intent, confidence, probabilities


def get_response(intent, intents):
    for item in intents["intents"]:
        if item["name"] == intent:
            return item["responses"][0]

    return "I don't understand that!"


if __name__ == "__main__":
    vectorizer, model, intents = train_model("svm")

    validation_data = load_validation_data()

    evaluate_model(model,vectorizer,validation_data)

    validate(
        model,
        vectorizer,
        validation_data
    )

    while True:
        text = input("You: ")

        if text.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        intent, confidence, probabilities = predict_intent(
            text,
            vectorizer,
            model
        )

        print("Probabilities:")

        for class_name, probability in zip(model.classes_, probabilities):
            print(f"{class_name}: {probability:.2f}")

        print(f"Prediction: {intent} ({confidence:.2f})")
        