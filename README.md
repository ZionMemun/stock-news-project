# 🚀 Stock News Sentiment Analysis Dashboard

An end-to-end data science project that collects stock-related news, analyzes sentiment using a machine learning model, and presents insights through an interactive dashboard.

---

## 📌 Overview

This project tracks financial news for selected stocks and classifies each article as:

- 🟢 Positive  
- ⚪ Neutral  
- 🔴 Negative  

It combines:

- 📡 Data collection (APIs & RSS)
- 🗄️ Database storage (SQLite)
- 🤖 Machine Learning (FinBERT)
- 📊 Dashboard visualization (Streamlit)

---

## ✨ Features

- 📰 Collect stock news from multiple sources  
- 🧠 Sentiment analysis using ML (FinBERT)  
- 🗃️ Store data in SQLite database  
- 🚫 Prevent duplicate entries (by URL)  
- 🔍 Advanced filtering (stock, sentiment, date)  
- 📈 Separate positive / negative analysis page  
- 🧾 Export data to CSV  
- 🗑️ Delete selected or all records  
- 🌙 Clean dark-themed dashboard  

---

## 🛠️ Tech Stack

- Python  
- SQLite  
- Pandas  
- Streamlit  
- PyTorch  
- Transformers (Hugging Face)  
- BeautifulSoup  
- Requests  

---

## 📂 Project Structure

```bash
stock_news_project/
│
├── app.py
├── dashboard.py
├── config.py
├── requirements.txt
│
├── collectors/
│   ├── finnhub_collector.py
│   └── google_rss.py
│
├── database/
│   └── db.py
│
├── ml/
│   └── predict.py
│
├── training/
│   └── train_finbert_clean_dataset.py
│
├── data/
│   └── news.db
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/stock-news-project.git
cd stock-news-project
```

## 2️⃣ Create virtual environment

### 🪟 Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 🍎 macOS / 🐧 Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Configure API keys

Edit `config.py`:

```python
FINNHUB_API_KEY = "your_api_key_here"

TRACKED_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
```

⚠️ Do NOT upload API keys to GitHub

---

# ▶️ Usage

## 📰 Step 1 — Collect News

```bash
python app.py
```

This will:
- Fetch news from APIs  
- Store in database  
- Run sentiment analysis  

## 📊 Step 2 — Launch Dashboard

```bash
streamlit run dashboard.py
```

Then open:

```
http://localhost:8501
```

---

## 🧠 How It Works

### 📡 Data Collection
- Google News RSS  
- Finnhub API  

### 🗄️ Database
- SQLite local storage  
- Unique URL constraint  
- Fast querying  

### 🤖 Sentiment Model
- FinBERT-based classifier  
- Input: title + summary  
- Output: Positive / Neutral / Negative  

### 📊 Dashboard
- Interactive filtering  
- Sentiment breakdown  
- Data export & management  

---

## 📋 Database Schema

### `news` table

| Column            | Description |
|------------------|------------|
| id               | Primary key |
| stock_symbol     | Stock ticker |
| source           | News source |
| title            | Article title |
| url              | Unique URL |
| summary          | Article summary |
| published_at     | Publish time |
| sentiment_label  | Sentiment |
| sentiment_score  | Confidence |
| collected_at     | Insert time |

---

## 🚀 Future Improvements

- ⏱️ Automatic scheduled collection  
- 📈 Stock price integration  
- 📊 Sentiment trend graphs  
- 🌐 Deploy dashboard online  
- 🧠 Improved ML model  

---

## 💡 Why This Project?

This project demonstrates:

- End-to-end data pipeline  
- Real-world NLP application  
- API integration  
- Database design  
- Dashboard development  

---

## 👤 Author

**Zion Memun**  
B.Sc. Data Science and Engineering Student  

---

## 📜 License

For educational and portfolio purposes.