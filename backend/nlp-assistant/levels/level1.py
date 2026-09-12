import json
import random

from levels.preprocessing import preprocess
from pathlib import Path

from levels.preprocessing import preprocess


BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_FILE = BASE_DIR / "data" / "intents.json"


def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)

def match_pattern(tokens, pattern_tokens):
    matched = 0
    input_index = 0

    for pattern in pattern_tokens:
        for i in range(input_index,len(tokens)):
            if tokens[i]==pattern:
                matched +=1
                input_index=i+1
                break

    return matched
def find_intent(text,intents):
    tokens = preprocess(text)

    best_intent = None
    best_score = 0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = preprocess(pattern)

            matched = match_pattern(tokens,pattern_tokens)

            if matched == len(pattern_tokens) or matched >= 2:

                score = matched + (
                    matched / len(pattern_tokens)
                )

                if score > best_score:
                    best_score = score
                    best_intent = intent

    if best_intent is None and len(tokens) <= 2:

        for intent in intents["intents"]:
            for keyword in intent["keywords"]:

                keyword_tokens = preprocess(keyword)

                if len(keyword_tokens) == 1 and keyword_tokens[0] in tokens:
                    best_intent = intent
                    break

            if best_intent is not None:
                break

    return best_intent

def generate_response(intent):
    return random.choice(intent['responses'])