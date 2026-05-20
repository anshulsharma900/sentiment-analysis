import streamlit as st
import pickle
import re
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analyser",
    page_icon="🐦",
    layout="centered"
)

# ── Load saved model & vectorizer ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = pickle.load(open('sentiment_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    return model, tfidf

model, tfidf = load_model()
stop_words = set(stopwords.words('english'))

# ── Text cleaning (same function used during training) ────────────────────────
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# ── Predict function ──────────────────────────────────────────────────────────
def predict(text):
    cleaned    = clean_tweet(text)
    vectorized = tfidf.transform([cleaned])
    label      = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    confidence = proba[list(model.classes_).index(label)] * 100
    return label, confidence

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🐦 Tweet Sentiment Analyser")
st.markdown("Type any tweet or sentence and the model will predict whether it's **positive** or **negative**.")
st.markdown("---")

# Single tweet input
st.subheader("Try a single tweet")
user_input = st.text_area(
    "Enter your text here:",
    placeholder="e.g. I absolutely love this new phone!",
    height=100
)

if st.button("Analyse", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, confidence = predict(user_input)

        if label == "positive":
            st.success(f"😊 **Positive** — Confidence: {confidence:.1f}%")
        else:
            st.error(f"😞 **Negative** — Confidence: {confidence:.1f}%")

        st.progress(int(confidence))

st.markdown("---")

# Batch — try multiple examples at once
st.subheader("Try example sentences")

examples = [
    "I absolutely love this phone, it's amazing!",
    "This movie was a complete waste of time.",
    "Just had the best meal of my life!",
    "I hate Mondays so much.",
    "The weather today is okay I guess.",
    "Worst customer service I have ever experienced.",
    "Feeling so grateful and happy today!",
    "My laptop keeps crashing, I'm done with this.",
]

if st.button("Run all examples"):
    results = []
    for ex in examples:
        label, conf = predict(ex)
        emoji = "😊" if label == "positive" else "😞"
        results.append({
            "Tweet": ex,
            "Sentiment": f"{emoji} {label.capitalize()}",
            "Confidence": f"{conf:.1f}%"
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(results), use_container_width=True)

st.markdown("---")

# About section
with st.expander("About this project"):
    st.markdown("""
    **Model:** Logistic Regression  
    **Text features:** TF-IDF (top 50,000 n-grams)  
    **Training data:** Sentiment140 — 1.6 million labelled tweets  
    **Accuracy:** ~78% on held-out test set  

    **Pipeline:**
    1. Clean text (remove links, mentions, punctuation)
    2. Convert words to numbers using TF-IDF
    3. Logistic Regression predicts positive or negative
    4. Confidence score from model probability

    Built with Python, Scikit-learn, NLTK, and Streamlit.
    """)
