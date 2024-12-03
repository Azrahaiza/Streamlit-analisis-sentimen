# Install required libraries (uncomment if needed)
# !pip install google-play-scraper wordcloud seaborn nltk scikit-learn streamlit matplotlib

# Import libraries
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from google_play_scraper import Sort, reviews
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the app ID for Info BMKG
app_id = "com.Info_BMKG"

# Fetch reviews for the app from Google Play Store
result, continuation_token = reviews(
    app_id,           # App ID
    lang='id',        # Language: Indonesian
    country='id',     # Country: Indonesia
    sort=Sort.NEWEST, # Sort by newest reviews
    count=2500,       # Fetch 2500 reviews
    filter_score_with=None  # No score filter
)

# Convert reviews to a DataFrame
df_busu = pd.DataFrame(result)

# Sentiment function based on the score
def sentiment(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

# Apply sentiment function to the score column
df_busu['sentiment'] = df_busu['score'].apply(sentiment)

# Streamlit app layout
st.title("Analisis Sentimen Aplikasi Info BMKG")

# Display the first 2500 data (original)
st.subheader("Tabel Data (2500 Review Original)")
st.dataframe(df_busu[['content', 'score', 'sentiment']].head(2500))

# Clean the text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", text)  # Remove unwanted characters
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

# Apply text cleaning
df_busu['text_clean'] = df_busu['content'].apply(clean_text)

# Remove stopwords
stop = set(stopwords.words('indonesian'))
df_busu['text_StopWord'] = df_busu['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Apply stemming
stemmer = PorterStemmer()
df_busu['text_stemmed'] = df_busu['text_StopWord'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Display the first 10 data after stemming
st.subheader("Tabel Data Setelah Stemming (10 Review)")
st.dataframe(df_busu[['content', 'text_stemmed', 'sentiment']].head(10))

# Plot Sentiment Distribution
st.subheader("Distribusi Sentimen")
fig, ax = plt.subplots()
sns.countplot(x='sentiment', data=df_busu, ax=ax)
ax.set_title('Sentiment Distribution')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
st.pyplot(fig)

# Prepare data for machine learning (TF-IDF Vectorization)
X = df_busu['text_stemmed']
y = df_busu['sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Support Vector Classifier (SVC) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = svm_model.predict(X_test_tfidf)

# Display Classification Report and Accuracy Score
st.subheader("Hasil Evaluasi Model")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
st.text(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Plot Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)

# Generate and Display Word Cloud
st.subheader("Word Cloud")
all_text = " ".join(df_busu["text_stemmed"].dropna())
wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(all_text)

fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)
