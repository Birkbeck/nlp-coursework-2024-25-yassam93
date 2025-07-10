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
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    # Filtering for purely alphabetic words to refine analysis.
    words = [w for w in words if w.isalpha()]
    

    syllables = sum(count_syl(w, d) for w in words)

    num_sentences = len(sentences)


    num_words = len(words)

    if num_sentences == 0 or num_words == 0:
        return 0.0

    asl = num_words / num_sentences

    asw = syllables / num_words

    return round(0.39 * asl + 11.8 * asw - 15.59, 4)


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        return len([phoneme for phoneme in d[word][0] if phoneme[-1].isdigit()])
    else:
        
        return len(re.findall(r'[aeiouy]+', word))




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
    """
    Parses the text of a DataFrame using spaCy, stores the parsed docs as a new column,
    and writes the resulting DataFrame to a pickle file.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column
        store_path (Path): Folder to save the pickle file
        out_name (str): Name of the pickle file
    
    Returns:
        pd.DataFrame: DataFrame with a new 'parsed' column
    """
    

    store_path.mkdir(parents=True, exist_ok=True)

    parsed_docs = []
    for text in df["text"]:
        if len(text) > nlp.max_length:
            chunks = [text[i:i+nlp.max_length] for i in range(0, len(text), nlp.max_length)]
            doc = None
            for chunk in chunks:
                subdoc = nlp(chunk)
                if doc is None:
                    doc = subdoc
                else:
                    doc = spacy.tokens.Doc.from_docs([doc, subdoc])
        else:
            doc = nlp(text)
        parsed_docs.append(doc)

    df["parsed"] = parsed_docs

    #saving parsed DataFrame to pickle
    pickle_path = store_path / out_name
    df.to_pickle(pickle_path)

    print(f"\n Parsed texts saved to: {pickle_path}")
    return df



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
    """Extracts the most common subjects of a given verb in a parsed document,
    ranked by Pointwise Mutual Information (PMI). Returns a list of tuples.
    
    Args:
        doc (spacy.tokens.Doc): Parsed document.
        target_verb (str): The verb to analyse (lemma form, e.g. 'hear').

    Returns:
        List[Tuple[str, float]]: Top 10 subjects ranked by PMI.
    """
    total_tokens = len(doc)
    if total_tokens == 0:
        return []

    verb_freq = 0
    subj_freq = Counter()
    joint_freq = Counter()

    for token in doc:
        if token.dep_ == "nsubj":
            subj_freq[token.text.lower()] += 1
        if token.lemma_ == target_verb and token.pos_ == "VERB":
            verb_freq += 1
            for child in token.children:
                if child.dep_ == "nsubj":
                    joint_freq[child.text.lower()] += 1

    if verb_freq == 0:
        return []


    pmi = {}

    for subj in joint_freq:
        p_xy = joint_freq[subj] / total_tokens
        p_x = subj_freq[subj] / total_tokens
        p_y = verb_freq / total_tokens
        if p_x > 0 and p_y > 0:
            pmi[subj] = math.log2(p_xy / (p_x * p_y))

    return sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:10]



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    for token in doc:
        if token.lemma_ == verb and token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "nsubj":
                    subjects.append(child.text.lower())
    return Counter(subjects).most_common(10)





def object_counts(doc):
    """Returns the 10 most common syntactic objects (dobj, pobj) in the parsed document."""
    objects = [
        token.text.lower()
        for token in doc
        if token.dep_ in ("dobj", "pobj") and token.is_alpha
    ]
    return Counter(objects).most_common(10)




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