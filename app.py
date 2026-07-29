from flask import Flask, render_template, request
import joblib
import nltk
import re
import string
from nltk.corpus import stopwords


# -------------------------------------------------
# Flask Application
# -------------------------------------------------

app = Flask(__name__)

# -------------------------------------------------
# Load Trained ML Model
# -------------------------------------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -------------------------------------------------
# Function : clean_text()
# Purpose  : Clean tweet before prediction
# -------------------------------------------------

def clean_text(text):
    """
    Clean raw tweet text for prediction.
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

    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = set()

    words = [
        word
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# -------------------------------------------------
# Home Route
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    emoji_icon = None

    if request.method == "POST":

        # User Input
        tweet = request.form["tweet"]

        # Clean Text
        cleaned = clean_text(tweet)

        # Convert into TF-IDF Vector
        vector = vectorizer.transform([cleaned])

        # Predict Sentiment
        prediction = model.predict(vector)[0]

        # Prediction Confidence
        probability = model.predict_proba(vector)

        confidence = round(
            max(probability[0]) * 100,
            2
        )

        # Emoji Mapping
        emoji = {
            "Positive": "😊",
            "Negative": "😠",
            "Neutral": "😐"
        }

        emoji_icon = emoji.get(
            prediction,
            "🙂"
        )

    return render_template(
        "index.html",
        prediction=prediction,
        sentiment_class=prediction.lower() if prediction else "",
        confidence=confidence,
        emoji=emoji_icon
    )


# -------------------------------------------------
# Run Application
# -------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
