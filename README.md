# 📰 Fake News Detection with Web Verification (Gradio + ML)

This project is a machine learning-powered web app that detects whether a news article is **Fake** or **Real**. It uses a Logistic Regression model trained on a labeled dataset and verifies the news by performing a real-time Google search using **SerpAPI**.

Built using:
- 💬 Natural Language Processing (TF-IDF)
- 🧠 Machine Learning (Logistic Regression)
- 🌐 SerpAPI (live web verification)
- 🖥️ Gradio (interactive user interface)

---

## 📚 Dataset

**Source**: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| Dataset | Description | Label |
|--------|-------------|-------|
| `Fake.csv` | Contains fake news articles | `0` |
| `True.csv` | Contains real news articles | `1` |

- Total records: ~44,000
- Only `text` column is used for prediction

---

## 🔧 Features

✅ Logistic Regression model trained on real/fake news  
✅ TF-IDF vectorization of text data  
✅ Real-time **Google snippet search** using **SerpAPI**  
✅ Cosine similarity to detect how similar your news is to real headlines  
✅ Simple and clean **Gradio** UI  
✅ Displays:
- ML Prediction (Fake or Real)
- Confidence Score
- Web Match Verification
- Similarity Score

---

## 🛠️ Installation

### 🔗 Requirements
- Python 3.8+
- Pip

### 📦 Install Dependencies

pip install -r requirements.txt

📥 Dataset Setup
Download Fake.csv and True.csv from Kaggle

Place them in the root project directory

🔑 SerpAPI Setup
Go to https://serpapi.com and create a free account

Get your API Key

In app.py, replace:
SERP_API_KEY = "your-serpapi-key"

 Run the App
python app.py
