import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#part a- Reading and preparing the dataset
def load_and_clean_data(csv_path):
    """
    Loads the dataset and applies all cleaning steps.
    """
    
    
    df = pd.read_csv(csv_path)

    #standardising the party label
    df['party'] = df['party'].replace({'Labour (Co-op)': 'Labour'})


    #removing entries where the speaker is listed like 'Speaker'
    df = df[df['party'] != 'Speaker']

    #keeping only the four most frequent parties
    top_parties = df['party'].value_counts().nlargest(4).index
    df = df[df['party'].isin(top_parties)]

    #retaining only actual speeches
    df = df[df['speech_class'] == 'Speech']


    #removing speeches that are too short to be meaningful
    df = df[df['speech'].str.len() >= 1000]

    return df

