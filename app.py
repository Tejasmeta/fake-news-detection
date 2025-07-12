import pandas as pd
import re
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from serpapi import GoogleSearch
import numpy as np

# ========== STEP 1: Load Dataset ==========
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df['label'] = 0
true_df['label'] = 1

df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ========== STEP 2: Preprocessing ==========
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df['text'] = df['text'].astype(str).apply(clean_text)

# ========== STEP 3: Vectorize ==========
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# ========== STEP 4: Train Model ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ========== STEP 5: SerpAPI Web Search ==========
SERP_API_KEY = "7698e8d463eb22ff3961f47a78e36122ef5e438af740918d51b9a9f6f105c5af"

def search_web(query):
    params = {
        "q": query,
        "num": 5,
        "api_key": SERP_API_KEY,
        "engine": "google",
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = []

    for res in results.get("organic_results", []):
        if "snippet" in res:
            snippets.append(res["snippet"])
    return snippets

# ========== STEP 6: Similarity Check ==========
def check_similarity(input_text, snippets):
    input_vec = vectorizer.transform([input_text])
    snippet_vecs = vectorizer.transform(snippets)
    sims = cosine_similarity(input_vec, snippet_vecs)
    return np.max(sims) if sims.size > 0 else 0

# ========== STEP 7: Prediction Logic ==========
def predict_news(news_text):
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned])
    ml_pred = model.predict(vec)[0]
    ml_conf = model.predict_proba(vec).max()

    try:
        snippets = search_web(news_text[:100])  # Use first 100 chars for query
        sim_score = check_similarity(cleaned, snippets)
    except Exception as e:
        sim_score = 0

    # Decision logic
    ml_result = "🟢 Real News" if ml_pred == 1 else "🔴 Fake News"
    web_result = "🔎 Match Found Online" if sim_score > 0.4 else "🚫 No Strong Match Online"

    return (
        f"{ml_result}\n"
        f"🧠 ML Confidence: {ml_conf:.2f}\n"
        f"{web_result}\n"
        f"🔁 Similarity Score: {sim_score:.2f}"
    )

# ========== STEP 8: Gradio UI ==========
description = (
    "📘 **Fake News Detection with Web Search Verification**\n\n"
    "- Trained on [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)\n"
    "- Uses Logistic Regression + TF-IDF to classify news\n"
    "- Verifies authenticity by comparing with real-time Google search results using SerpAPI\n\n"
    "🧪 Paste any news article or headline below to test!"
)

iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(lines=10, placeholder="Paste the news article here..."),
    outputs="text",
    title="📰 Fake News Detector + Web Verifier",
    description=description
)

iface.launch()
