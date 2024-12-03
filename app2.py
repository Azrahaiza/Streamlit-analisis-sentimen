import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import PorterStemmer
import seaborn as sns
from google_play_scraper import Sort, reviews

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

st.title("Analisis Sentimen Ulasan Aplikasi Info BMKG")

# Fetch data
@st.cache_data
def fetch_reviews(app_id, count=100):
    result, _ = reviews(
        app_id,
        lang='id',
        country='id',
        sort=Sort.NEWEST,
        count=count
    )
    df = pd.DataFrame(np.array(result), columns=['review'])
    return df.join(pd.DataFrame(df.pop('review').tolist()))

app_id = "com.Info_BMKG"
st.sidebar.header("Konfigurasi")
review_count = st.sidebar.slider("Jumlah ulasan yang diambil", min_value=100, max_value=5000, value=1000, step=100)

df = fetch_reviews(app_id, review_count)

# Display raw data
st.subheader("Data Mentah")
st.dataframe(df)

# Sentiment classification
def sentiment(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

df['sentiment'] = df['score'].apply(sentiment)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df['text_clean'] = df['content'].apply(clean_text)

# Remove stopwords
stop = stopwords.words('indonesian')
df['text_stopwords_removed'] = df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Stemming
stemmer = PorterStemmer()
df['text_stemmed'] = df['text_stopwords_removed'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Sentiment distribution visualization
st.subheader("Distribusi Sentimen")
sns.countplot(x='sentiment', data=df)
st.pyplot()

# Word Cloud
st.subheader("Word Cloud")
all_text = " ".join(df["text_stemmed"].dropna())
wordcloud = WordCloud(width=700, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot()

# Export data
st.sidebar.subheader("Ekspor Data")
if st.sidebar.button("Unduh CSV"):
    df.to_csv("data_reviews_with_sentiment.csv", index=False)
    st.sidebar.success("Data berhasil diekspor!")
