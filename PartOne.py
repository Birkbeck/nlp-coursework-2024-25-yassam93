#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import pandas as pd
import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('cmudict')
#nltk.download('stopwords')

import spacy
import string
from pathlib import Path

import re
from nltk.tokenize import word_tokenize, sent_tokenize

from collections import Counter
import math

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000




def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    pass


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    pass



def read_novels(path=Path.cwd() / "texts" / "novels"):
    """
    Reads all .txt files in the novels directory and returns a DataFrame with columns:
    'text', 'title', 'author', 'year'. The DataFrame is sorted by year (ascending).
    """

    data = []

    for file in path.glob("*.txt"):
        try:
            title, author, year = file.stem.split("-")
            year = int(year)
        except ValueError:
            print(f"Skipping file due to naming error: {file.name}")
            continue

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        data.append({
            "text": text,
            "title": title.strip(),
            "author": author.strip(),
            "year": year
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by="year").reset_index(drop=True)
    return df




def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    pass



def nltk_ttr(text):
    """
    Calculates the Type-Token Ratio (TTR) of a given text using NLTK.
    Ignores punctuation and is case-insensitive.
    """
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t not in string.punctuation]
    return len(set(words)) / len(words) if words else 0.0


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    pass



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    #path = Path.cwd() / "p1-texts" / "novels"
    #print(path)
    #df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    #print(df.head())
    #nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    #print(get_ttrs(df))
    #print(get_fks(df))
    #df = pd.read_pickle(Path.cwd() / "pickles" /"name.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """