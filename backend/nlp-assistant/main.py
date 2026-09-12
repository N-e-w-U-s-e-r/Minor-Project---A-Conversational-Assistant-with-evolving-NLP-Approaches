import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAINED_MODELS = BASE_DIR / "trained_models"

from levels.level1 import (
    load_intents,
    find_intent,
    generate_response
)

from levels.level2 import (
    predict_intent as predict_level2_intent,
    get_response as get_level2_response
)

from levels.level3 import (
    predict_intent as predict_level3_intent,
    get_response as get_level3_response
)

from levels.level4 import (
    predict_intent as predict_level4_intent,
    get_response as get_level4_response
)


# ============================================================
# LEVEL 1
# ============================================================

level1_intents = load_intents()


# ============================================================
# LEVEL 2 - LOAD TRAINED MODEL
# ============================================================


level2_vectorizer = joblib.load(
    TRAINED_MODELS / "level2_vectorizer.joblib"
)

level2_model = joblib.load(
    TRAINED_MODELS / "level2_model.joblib"
)

level2_training_vectors = joblib.load(
    TRAINED_MODELS / "level2_training_vectors.joblib"
)

level2_intents = joblib.load(
    TRAINED_MODELS / "level2_intents.joblib"
)

level3_vectorizer = joblib.load(
    TRAINED_MODELS / "level3_vectorizer.joblib"
)

level3_model = joblib.load(
    TRAINED_MODELS / "level3_model.joblib"
)

level3_intents = joblib.load(
    TRAINED_MODELS / "level3_intents.joblib"
)
# ============================================================
# LEVEL 1
# ============================================================

def run_level1(message):

    intent = find_intent(
        message,
        level1_intents
    )

    if intent:

        return {
            "level": 1,
            "intent": intent["name"],
            "response": generate_response(intent)
        }

    return {
        "level": 1,
        "intent": "unknown",
        "response": "I don't understand that!"
    }


# ============================================================
# LEVEL 2
# ============================================================

def run_level2(message):

    intent, confidence, similarity, combined_score = (
        predict_level2_intent(
            message,
            level2_vectorizer,
            level2_model,
            level2_training_vectors
        )
    )

    if intent == "unknown":

        response = "I don't understand that!"

    else:

        response = get_level2_response(
            intent,
            level2_intents
        )

    return {
        "level": 2,
        "intent": intent,
        "confidence": float(confidence),
        "similarity": float(similarity),
        "combined_score": float(combined_score),
        "response": response
    }


# ============================================================
# LEVEL 3
# ============================================================

# def run_level3(message):

#     intent, confidence, probabilities = (
#         predict_level3_intent(
#             message,
#             level3_vectorizer,
#             level3_model
#         )
#     )

#     if intent == "unknown":

#         response = "I don't understand that!"

#     else:

#         response = get_level3_response(
#             intent,
#             level3_intents
#         )

#     return {
#         "level": 3,
#         "intent": intent,
#         "confidence": float(confidence),
#         "response": response
#     }

def run_level3(message):

    intent, confidence, probabilities = (
        predict_level3_intent(
            message,
            level3_vectorizer,
            level3_model
        )
    )

    print("LEVEL 3 DEBUG")
    print("Message:", message)
    print("Predicted intent:", intent)
    print("Confidence:", confidence)

    if intent == "unknown":
        response = "I don't understand that!"
    else:
        response = get_level3_response(
            intent,
            level3_intents
        )

    print("Response:", response)

    return {
        "level": 3,
        "intent": intent,
        "confidence": float(confidence),
        "response": response
    }
# ============================================================
# LEVEL 4
# ============================================================

def run_level4(message):

    intent, confidence, result = (
        predict_level4_intent(message)
    )

    response = get_level4_response(intent)

    return {
        "level": 4,
        "intent": intent,
        "confidence": float(confidence),
        "response": response
    }


# ============================================================
# MAIN NLP INTERFACE
# ============================================================

def process_message(message, level):

    if level == 1:
        return run_level1(message)

    if level == 2:
        return run_level2(message)

    if level == 3:
        return run_level3(message)

    if level == 4:
        return run_level4(message)

    raise ValueError("Invalid NLP level. Choose 1, 2, 3, or 4.")


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():

    print("NLP Assistant is ready.")

    while True:

        level_input = input(
            "\nEnter NLP level (1-4) or 'quit': "
        )

        if level_input.lower() == "quit":
            print("Assistant: Goodbye!")
            break

        try:
            level = int(level_input)
        except ValueError:
            print("Please enter a number from 1 to 4.")
            continue

        if level not in (1, 2, 3, 4):
            print("Please enter a number from 1 to 4.")
            continue

        message = input("You: ")

        result = process_message(
            message,
            level
        )

        print(
            f"Level: {result['level']}"
        )

        print(
            f"Intent: {result['intent']}"
        )

        if "confidence" in result:
            print(
                f"Confidence: "
                f"{result['confidence']:.2f}"
            )

        if "similarity" in result:
            print(
                f"Similarity: "
                f"{result['similarity']:.2f}"
            )

        if "combined_score" in result:
            print(
                f"Combined score: "
                f"{result['combined_score']:.2f}"
            )

        print(
            f"Assistant: {result['response']}"
        )


if __name__ == "__main__":
    main()