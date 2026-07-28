import re
import string
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ==========================================================
# Download Required NLTK Resources (Runs Only Once)
# ==========================================================

nltk.download("stopwords")


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(
    "dataset/twitter_training.csv",
    header=None
)

df.columns = [
    "id",
    "topic",
    "sentiment",
    "text"
]


# ==========================================================
# Data Preprocessing
# ==========================================================

# Keep only required columns
df = df[["text", "sentiment"]]

# Remove missing values
df.dropna(inplace=True)

# Remove irrelevant tweets
df = df[df["sentiment"] != "Irrelevant"]


# ==========================================================
# Text Cleaning Function
# ==========================================================

def clean_text(text):
    """
    Clean tweet text for Machine Learning.

    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove mentions
    - Remove hashtags
    - Remove numbers
    - Remove punctuation
    - Remove extra spaces
    - Remove English stopwords

    Returns:
        Cleaned tweet text
    """

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    stop_words = set(stopwords.words("english"))

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# ==========================================================
# Apply Text Cleaning
# ==========================================================

df["clean_text"] = df["text"].apply(clean_text)


# ==========================================================
# Dataset Overview
# ==========================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nSample Record:")
print(df[["text", "clean_text", "sentiment"]].head(1))


# ==========================================================
# TF-IDF Feature Extraction
# ==========================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

print("\nTF-IDF Feature Matrix:")
print(X.shape)


# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])


# ==========================================================
# Train Machine Learning Model
# ==========================================================

print("\nTraining Naive Bayes Model...")

model = MultinomialNB()

model.fit(X_train, y_train)

print("Training Completed Successfully.")


# ==========================================================
# Model Evaluation
# ==========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ==========================================================
# Save Model
# ==========================================================

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nSaved Files")
print("✔ model.pkl")
print("✔ vectorizer.pkl")

print("\nProject Completed Successfully.")