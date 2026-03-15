import re


def tokenize(text):
    return re.findall(r"\b[a-z]+\b", text.lower())
