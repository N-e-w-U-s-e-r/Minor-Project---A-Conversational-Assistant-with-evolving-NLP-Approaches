import json
import time
import os
import random
import re
from pathlib import Path

from google import genai


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_FILE = BASE_DIR / "data" / "intents.json"

MODEL_NAME = "gemini-3.5-flash-lite"

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set the environment variable before running Level 4."
    )

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# 2. Load intents
# --------------------------------------------------

with open(INTENTS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

INTENTS = [
    item["name"]
    for item in data["intents"]
]

if "unknown" not in INTENTS:
    INTENTS.append("unknown")


# --------------------------------------------------
# 3. Gemini intent classification
# --------------------------------------------------



def predict_intent(text):
    start_time = time.perf_counter()
    confidence=0.0
    result={}

    prompt = f"""
Classify the user's message into exactly one of these intents:
{", ".join(INTENTS)}

Rules:
- Return only one intent name from the list.
- Do not explain your answer.
- If no intent matches, return unknown.

User message: {text}
Intent:
"""

    try:
        response = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            generation_config={
                "thinking_level": "minimal",
                "max_output_tokens": 32,
            },
        )

        raw_response = (response.output_text or "").strip()
        predicted_intent = raw_response.strip().lower()

        # Accept only known intent names.
        valid_intents = {intent.lower(): intent for intent in INTENTS}

        if predicted_intent in valid_intents:
            predicted_intent = valid_intents[predicted_intent]
            confidence=1.0
        else:
            predicted_intent = "unknown"
            confidence=0.0

        return predicted_intent, confidence, result

    except Exception as error:
        print(f"Gemini classification error: {error}")
        return "unknown"

    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Gemini response time: {elapsed:.2f} seconds")

# --------------------------------------------------
# 4. Generate a response for the predicted intent
# --------------------------------------------------

def get_response(intent):
    for item in data["intents"]:
        if item["name"] == intent:
            return random.choice(item["responses"])

    return "I'm sorry, I don't understand."


# --------------------------------------------------
# 5. Run Level 4 independently
# --------------------------------------------------

if __name__ == "__main__":
    print("Level 4: Gemini Intent Classifier")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("You: ")

            if text.strip().lower() == "quit":
                print("Assistant: Goodbye!")
                break

            intent, confidence, result = predict_intent(text)

            print(f"Prediction: {intent}")
            print(f"Assistant: {get_response(intent)}")

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break
