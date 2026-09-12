
import json
import re
from difflib import SequenceMatcher

from llama_cpp import Llama


INTENTS_FILE = "data/intents.json"

MODEL_PATH = "models/Qwen3-4B-Q5_K_M.gguf"

MAX_RETRIES = 5

MIN_PATTERN_WORDS = 3
MIN_RESPONSE_WORDS = 2

SIMILARITY_THRESHOLD = 0.80

GENERIC_INTENT_NAMES = {
    "travel",
    "shopping",
    "movie",
    "movies",
    "food",
    "restaurant",
    "music",
    "sports",
    "technology",
    "business",
    "help",
    "information",
    "question",
    "request",
    "general",
}


def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_existing_intents(data):
    return [
        intent["name"]
        for intent in data["intents"]
    ]


def get_existing_patterns(data):
    patterns = []

    for intent in data["intents"]:
        patterns.extend(intent["patterns"])

    return patterns


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def similarity(text1, text2):
    return SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2)
    ).ratio()


def create_prompt(
    existing_intents,
    rejected_intents=None,
    previous_error=None
):
    existing = ", ".join(existing_intents)

    rejected_section = ""

    if rejected_intents:
        rejected = ", ".join(rejected_intents)

        rejected_section = f"""
Previously rejected/generated intent names:

{rejected}

Do NOT generate any of these intent names again.
"""

    error_section = ""

    if previous_error:
        error_section = f"""
The previous generation attempt failed because:

{previous_error}

Correct this problem in your new response.
"""

    return f"""
/no_think
You are generating ONE training intent for a conversational chatbot.

Existing intents:
{existing}

{rejected_section}

{error_section}

Generate ONE NEW intent that represents ONE specific and clearly defined
user goal.

IMPORTANT:

The intent must NOT be a broad category.

Bad examples:
- travel
- shopping
- movie
- restaurant
- music
- food
- technology

Good examples:
- book_flight
- cancel_flight
- track_order
- request_refund
- restaurant_recommendation
- movie_recommendation
- check_flight_status

The generated intent must be clearly different from every existing intent.

Every pattern MUST express the SAME underlying user intention.

For example, if the intent is "book_flight", all patterns should be about
booking a flight. Do not mix booking, cancellation, flight status,
baggage, or general travel questions.

Patterns should differ in wording while keeping the same meaning.

Requirements:

- name: short lowercase snake_case intent name
- patterns: exactly 5 natural messages that a USER would say
- keywords: 3 to 5 relevant keywords
- responses: 2 to 4 natural chatbot responses
- Do not duplicate an existing intent.
- Do not duplicate a previously rejected/generated intent.
- Do not create a broad category.
- Do not generate greeting, goodbye, thanks, or weather.
- Do not combine multiple unrelated user goals.
- Patterns must sound like real user messages.
- Patterns must all express the same specific intent.
- Responses must directly address that specific intent.
- Do not describe what a user might say.
- Write the actual user messages.
- Do not use "example" as the intent name.
- Avoid vague intent names such as "help", "information", "request",
  "travel", "shopping", "movie", "restaurant", or "general".
- Prefer a specific action, request, or information need.

OUTPUT FORMAT:

Return ONLY one valid JSON object.

The response MUST begin with "{{" and end with "}}".

Do not include any text before the JSON object.
Do not include any text after the JSON object.
Do not include markdown.
Do not include ```json.
Do not include ``` .
Do not include explanations.

Use exactly this structure:

{{
    "name": "specific_intent_name",
    "patterns": [
        "user message 1",
        "user message 2",
        "user message 3",
        "user message 4",
        "user message 5"
    ],
    "keywords": [
        "keyword1",
        "keyword2",
        "keyword3"
    ],
    "responses": [
        "response 1",
        "response 2"
    ]
}}
"""


def generate_one_intent(
    model,
    existing_intents,
    rejected_intents=None,
    previous_error=None
):
    prompt = create_prompt(
        existing_intents,
        rejected_intents,
        previous_error
    )

    result = model(
        prompt,
        max_tokens=1200,
        temperature=0.3,
        top_p=0.9,
        stop=["```"]
    )

    return result["choices"][0]["text"]


def extract_json(text):
    decoder = json.JSONDecoder()

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in model output."
        )

    try:
        data, _ = decoder.raw_decode(
            text[start:]
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON output: {error}"
        ) from error

    return data


def validate_structure(
    intent,
    existing_intents
):
    required_fields = {
        "name",
        "patterns",
        "keywords",
        "responses"
    }

    if not isinstance(intent, dict):
        raise ValueError(
            "Generated intent is not a JSON object."
        )

    missing = required_fields - set(intent.keys())

    if missing:
        raise ValueError(
            f"Missing fields: {missing}"
        )

    name = intent["name"]

    if not isinstance(name, str):
        raise ValueError(
            "Intent name must be a string."
        )

    name = normalize_text(name).replace(" ", "_")

    if not name:
        raise ValueError(
            "Intent name is empty."
        )

    if name in existing_intents:
        raise ValueError(
            f"Duplicate existing intent: {name}"
        )

    if name in GENERIC_INTENT_NAMES:
        raise ValueError(
            f"Intent name is too broad: {name}"
        )

    if not isinstance(intent["patterns"], list):
        raise ValueError(
            "Patterns must be a list."
        )

    if not isinstance(intent["keywords"], list):
        raise ValueError(
            "Keywords must be a list."
        )

    if not isinstance(intent["responses"], list):
        raise ValueError(
            "Responses must be a list."
        )

    if len(intent["patterns"]) != 5:
        raise ValueError(
            "Intent must contain exactly 5 patterns."
        )

    if not 3 <= len(intent["keywords"]) <= 5:
        raise ValueError(
            "Intent must contain between 3 and 5 keywords."
        )

    if not 2 <= len(intent["responses"]) <= 4:
        raise ValueError(
            "Intent must contain between 2 and 4 responses."
        )

    for pattern in intent["patterns"]:

        if not isinstance(pattern, str):
            raise ValueError(
                "Every pattern must be a string."
            )

        if len(pattern.split()) < MIN_PATTERN_WORDS:
            raise ValueError(
                f"Pattern is too short: '{pattern}'"
            )

    for keyword in intent["keywords"]:

        if not isinstance(keyword, str):
            raise ValueError(
                "Every keyword must be a string."
            )

        if not keyword.strip():
            raise ValueError(
                "Empty keyword found."
            )

    for response in intent["responses"]:

        if not isinstance(response, str):
            raise ValueError(
                "Every response must be a string."
            )

        if len(response.split()) < MIN_RESPONSE_WORDS:
            raise ValueError(
                f"Response is too short: '{response}'"
            )

    intent["name"] = name


def check_pattern_duplicates(patterns):

    normalized = [
        normalize_text(pattern)
        for pattern in patterns
    ]

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Duplicate patterns found."
        )

    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):

            score = similarity(
                normalized[i],
                normalized[j]
            )

            if score >= SIMILARITY_THRESHOLD:
                raise ValueError(
                    "Two generated patterns are too similar."
                )


def check_response_duplicates(responses):

    normalized = [
        normalize_text(response)
        for response in responses
    ]

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Duplicate responses found."
        )

    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):

            score = similarity(
                normalized[i],
                normalized[j]
            )

            if score >= SIMILARITY_THRESHOLD:
                raise ValueError(
                    "Two generated responses are too similar."
                )


def check_existing_pattern_similarity(
    patterns,
    existing_patterns
):

    for pattern in patterns:

        for existing in existing_patterns:

            score = similarity(
                pattern,
                existing
            )

            if score >= SIMILARITY_THRESHOLD:
                raise ValueError(
                    f"Generated pattern is too similar to "
                    f"an existing pattern: '{pattern}'"
                )


def validate_quality(
    intent,
    existing_patterns
):
    check_pattern_duplicates(
        intent["patterns"]
    )

    check_response_duplicates(
        intent["responses"]
    )

    check_existing_pattern_similarity(
        intent["patterns"],
        existing_patterns
    )


def validate_intent(
    intent,
    existing_intents,
    existing_patterns
):
    validate_structure(
        intent,
        existing_intents
    )

    validate_quality(
        intent,
        existing_patterns
    )


def generate_intents(
    model,
    existing_intents,
    existing_patterns,
    number_of_intents
):

    generated_intents = []

    while len(generated_intents) < number_of_intents:

        current_intents = (
            existing_intents
            + [
                intent["name"]
                for intent in generated_intents
            ]
        )

        current_patterns = (
            existing_patterns
            + [
                pattern
                for intent in generated_intents
                for pattern in intent["patterns"]
            ]
        )

        print(
            f"\nGenerating intent "
            f"{len(generated_intents) + 1}"
            f"/{number_of_intents}..."
        )

        accepted = False

        rejected_intents = []
        previous_error = None

        for attempt in range(1, MAX_RETRIES + 1):

            try:

                raw_output = generate_one_intent(
                    model,
                    current_intents,
                    rejected_intents,
                    previous_error
                )

                intent = extract_json(
                    raw_output
                )

                validate_intent(
                    intent,
                    current_intents,
                    current_patterns
                )

                generated_intents.append(
                    intent
                )

                print(
                    f"Accepted: {intent['name']}"
                )

                accepted = True
                break

            except (
                ValueError,
                json.JSONDecodeError
            ) as error:

                error_message = str(error)

                print(
                    f"Attempt {attempt} failed: "
                    f"{error_message}"
                )

                previous_error = error_message

                # If a JSON object was produced, try to
                # remember its name so the next attempt
                # does not repeat it.
                try:
                    failed_intent = extract_json(
                        raw_output
                    )

                    failed_name = failed_intent.get(
                        "name"
                    )

                    if isinstance(
                        failed_name,
                        str
                    ):
                        failed_name = (
                            normalize_text(
                                failed_name
                            ).replace(" ", "_")
                        )

                        if failed_name:
                            rejected_intents.append(
                                failed_name
                            )

                except (
                    ValueError,
                    json.JSONDecodeError,
                    UnboundLocalError
                ):
                    pass

        if not accepted:

            print(
                "\nCould not generate a valid "
                "intent after several attempts."
            )

            break

    return generated_intents


def save_intents(data, generated_intents):
    """Append generated intents to intents.json."""
    data["intents"].extend(generated_intents)

    with open(INTENTS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nSaved {len(generated_intents)} new intent(s) to {INTENTS_FILE}.")


def main():

    print("Loading existing intents...")

    data = load_intents()

    existing_intents = get_existing_intents(data)
    existing_patterns = get_existing_patterns(data)

    print("\nExisting intents:")

    for intent in existing_intents:
        print(f"- {intent}")

    number_of_intents = int(
        input(
            "\nHow many new intents should be generated? "
        )
    )

    print("\nLoading Qwen3-4B...")

    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        verbose=False
    )

    generated_intents = generate_intents(
        model,
        existing_intents,
        existing_patterns,
        number_of_intents
    )

    if not generated_intents:

        print(
            "\nNo valid intents were generated."
        )

        return

    print("\n" + "=" * 60)
    print("GENERATED INTENTS")
    print("=" * 60)

    print(
        json.dumps(
            {
                "intents": generated_intents
            },
            indent=4,
            ensure_ascii=False
        )
    )

    # Save only successfully validated intents
    save_intents(
        data,
        generated_intents
    )

    print("\nGeneration and saving complete.")


if __name__ == "__main__":
    main()

