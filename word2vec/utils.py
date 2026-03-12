import re


def tokenize(text):
    """
    Very simple tokenizer.
    Only keeps alphabetic words and lowercases everything.
    """
    return re.findall(r"\b[a-z]+\b", text.lower())
