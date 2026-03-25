import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 Spam Email Detection")

# Dataset
data = {
    "text": [
        "Win money now!!!",
        "Hello, how are you?",
        "Claim your prize now",
        "Let's meet tomorrow",
        "Free entry in lottery",
        "Are you coming today?",
        "Congratulations you won a car",
        "Call me when you are free"
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)

# User input
msg = st.text_area("Enter your email/message:")

# Button
if st.button("Check"):
    if msg:
        msg_vec = vectorizer.transform([msg])
        result = model.predict(msg_vec)[0]
        
        if result == "spam":
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT Spam (Ham)")
    else:
        st.warning("⚠️ Please enter a message")