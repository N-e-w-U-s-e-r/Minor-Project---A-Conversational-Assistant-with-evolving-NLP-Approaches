from levels.level1 import load_intents, find_intent


def test_level1():
    intents = load_intents()

    test_cases = [
        ("hello", "greeting"),
        ("hellllo", "greeting"),
        ("good morning", "greeting"),

        ("goodbye", "goodbye"),
        ("see you later", "goodbye"),

        ("thank you", "thanks"),
        ("thanks a lot", "thanks"),

        ("what is the weather", "weather"),
        ("what is weather", "weather"),
        ("weather", "weather"),
        ("hi what is weather", "weather"),

        ("weather is terrible", None),
        ("hhchchfc", None),
    ]

    passed = 0

    for text, expected in test_cases:
        intent = find_intent(text, intents)

        actual = intent["name"] if intent else None

        if actual == expected:
            print(f"PASS: {text}")
            passed += 1
        else:
            print(
                f"FAIL: {text} | "
                f"Expected: {expected} | "
                f"Got: {actual}"
            )

    print()
    print(f"Passed: {passed}/{len(test_cases)}")


if __name__ == "__main__":
    test_level1()