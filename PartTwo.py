
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score



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

#part b–Vectorising and splitting the dataset
def vectorise_and_split(df):
    """
    Vectorises the speech text and splits data for training and testing.
    """
    print(f"\nDataset shape after cleaning: {df.shape}")


    vectoriser = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )

    X = vectoriser.fit_transform(df['speech'])
    y = df['party']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=26
    )

    return X_train, X_test, y_train, y_test, vectoriser



#run loading and vectorisation
if __name__ == "__main__":
    csv_path = Path.cwd() / "texts" / "speeches" / "hansard40000.csv"

    df = load_and_clean_data(csv_path)
    X_train, X_test, y_train, y_test, vectoriser = vectorise_and_split(df)

    #part C–Train models and review performance
    print("\nTraining Random Forest (300 trees)...")
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=26)
    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_test)

    print("\nRandom Forest results:")

    print(classification_report(y_test, rf_preds, digits=4))
    print("Macro-average F1 score:", round(f1_score(y_test, rf_preds, average='macro'), 4))

    print("\nTraining Support Vector Machine (linear kernel)...")
    
    svm = SVC(kernel='linear', class_weight='balanced', random_state=26)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)

    print("\nSupport Vector Machine results:")
    
    print(classification_report(y_test, svm_preds, digits=4))
    print("Macro-average F1 score:", round(f1_score(y_test, svm_preds, average='macro'), 4))

    #part D– Re-vectorising with n-grams
    print("\nNow re-vectorising using unigrams, bigrams and trigrams...")


    vectoriser_ngrams = TfidfVectorizer(
        stop_words='english',
        max_features=3000,
        ngram_range=(1, 3),
        lowercase=True
    )

    X = vectoriser_ngrams.fit_transform(df["speech"])
   
   
    y = df["party"]

    X_train_ng, X_test_ng, y_train_ng, y_test_ng = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=26
    )


    print("\nTraining Random Forest with n-gram features...")
    
    rf_ng = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=26)
    rf_ng.fit(X_train_ng, y_train_ng)
    rf_preds_ng = rf_ng.predict(X_test_ng)

    print("\nRandom Forest (n-grams) results:")
    
    print(classification_report(y_test_ng, rf_preds_ng, digits=4))

    print("\nTraining Support Vector Machine with n-gram features...")
    svm_ng = SVC(kernel='linear', class_weight='balanced', random_state=26)
    svm_ng.fit(X_train_ng, y_train_ng)
    svm_preds_ng = svm_ng.predict(X_test_ng)

    print("\nSupport Vector Machine (n-grams) results:")
    print(classification_report(y_test_ng, svm_preds_ng, digits=4))
    print("Macro-average F1 score:", round(f1_score(y_test_ng, svm_preds_ng, average='macro'), 4))




#part E–Last Enhancement with Custom Tokeniser and Vectorisation

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

#NLTKs needed:

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

#already downloaded in Part 1:
# nltk.download('punkt')
# nltk.download('stopwords')


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def lemmatising_tokeniser(text):
    """
    Tokeniser that:
    - converts text to lowercase
    - removes stopwords and punctuation
    - and applies WordNet lemmatisation

    Based on techniques demonstrated in the lab exercises.
    """

    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(token) for token in tokens
        if token.isalpha() and token not in stop_words
    ]

    return tokens

print("\nPart E – Improving Classification with a Custom Tokeniser")
print("Preparing TF-IDF vectoriser with n-grams and lemmatisation...")

vectoriser = TfidfVectorizer(
    tokenizer=lemmatising_tokeniser,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=5,
    max_df=0.9,
    max_features=3000
)

X = vectoriser.fit_transform(df["speech"])

y = df["party"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=26
)

#train and evaluate Linear SVM with a range of C values

print("\nTraining and evaluating the Linear SVM model using refined features...")

for c in [0.1, 0.5, 1, 5, 10]:
    print(f"\nEvaluating Linear SVM with regularisation parameter C = {c}")
    svm = SVC(kernel='linear', C=c, class_weight='balanced', random_state=26)
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)



    print(f"\nClassification results for SVM (C = {c}):")
    print(classification_report(y_test, predictions, digits=4))
    f1 = f1_score(y_test, predictions, average='macro')
    print(f"Macro-average F1 score: {f1:.4f}")


#complement Naive Bayes for comparison

print("\nNow evaluating the Complement Naive Bayes model...")

nb = ComplementNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_test)

print("\nNaive Bayes classification results:")

print(classification_report(y_test, nb_preds, digits=4))
f1_nb = f1_score(y_test, nb_preds, average='macro')
print(f"Macro-average F1 score: {f1_nb:.4f}")


