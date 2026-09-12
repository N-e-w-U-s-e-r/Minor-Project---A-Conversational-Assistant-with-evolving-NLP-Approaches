import json
import random

from transformers import pipeline


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_FILE = BASE_DIR / "data" / "intents.json"

with open(INTENTS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)


INTENTS = [
    intent["name"]
    for intent in data["intents"]
]


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


def predict_intent(text):
    result = classifier(
        text,
        candidate_labels=INTENTS
    )

    intent = result["labels"][0]
    confidence = result["scores"][0]

    return intent, confidence, result


def get_response(intent):
    for item in data["intents"]:
        if item["name"] == intent:
            return random.choice(item["responses"])

    return "I'm sorry, I don't understand."


if __name__ == "__main__":
    while True:
        text = input("You: ")

        if text.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        intent, confidence, result = predict_intent(text)

        response = get_response(intent)

        print("Probabilities:")

        for label, score in zip(
            result["labels"],
            result["scores"]
        ):
            print(f"{label}: {score:.2f}")

        print(f"Prediction: {intent} ({confidence:.2f})")
        print(f"Assistant: {response}")