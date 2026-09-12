
import os
import joblib

from levels.level2 import train_model as train_level2
from levels.level3 import train_model as train_level3


MODEL_DIR = "trained_models"


def train_level2_model():
    print("Training Level 2...")

    (
        vectorizer,
        model,
        training_vectors,
        labels,
        intents
    ) = train_level2()

    joblib.dump(
        vectorizer,
        f"{MODEL_DIR}/level2_vectorizer.joblib"
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/level2_model.joblib"
    )

    joblib.dump(
        training_vectors,
        f"{MODEL_DIR}/level2_training_vectors.joblib"
    )

    joblib.dump(
        labels,
        f"{MODEL_DIR}/level2_labels.joblib"
    )

    joblib.dump(
        intents,
        f"{MODEL_DIR}/level2_intents.joblib"
    )

    print("Level 2 saved.")


def train_level3_model():
    print("Training Level 3...")

    vectorizer, model, intents = train_level3("svm")

    joblib.dump(
        vectorizer,
        f"{MODEL_DIR}/level3_vectorizer.joblib"
    )

    joblib.dump(
        model,
        f"{MODEL_DIR}/level3_model.joblib"
    )

    joblib.dump(
        intents,
        f"{MODEL_DIR}/level3_intents.joblib"
    )

    print("Level 3 saved.")


def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    train_level2_model()

    train_level3_model()

    print("\nAll models trained and saved.")


if __name__ == "__main__":
    main()

