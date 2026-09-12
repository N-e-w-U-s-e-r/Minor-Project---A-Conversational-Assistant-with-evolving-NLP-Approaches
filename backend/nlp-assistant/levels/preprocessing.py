import re

def preprocess(text):
    text = text.lower()
    text = text.strip()

    text = re.sub(r"(.)\1{2,}",r"\1\1",text)
    text = re.sub(r"[^\w\s]","",text)

    tokens = text.split()
    return tokens