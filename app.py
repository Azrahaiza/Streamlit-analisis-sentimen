import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from google_play_scraper import reviews, Sort

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
    count=5000,       # Number of reviews to fetch
    filter_score_with=None  # No score filter
)

# Convert reviews to a DataFrame
df_busu = pd.DataFrame(result, columns=['review'])
df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist(), columns=['content', 'score', 'at', 'userName']))

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

# Clean the text data
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()  # Convert to lowercase
    # Remove usernames, URLs, emoticons, punctuation, non-alphanumeric characters, and numbers
    df[new_text_field_name] = df[new_text_field_name].apply(
        lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
    )
    # Remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    return df

# Apply text cleaning
df_busu_clean = clean_text(df_busu, 'content', 'text_clean')

# Remove stopwords
stop = stopwords.words('indonesian')
df_busu_clean['text_StopWord'] = df_busu_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Apply stemming
stemmer = PorterStemmer()
df_busu_clean['text_stemmed'] = df_busu_clean['text_StopWord'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Prepare data for machine learning (TF-IDF Vectorization)
X = df_busu_clean['text_stemmed']
y = df_busu_clean['sentiment']

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

# Streamlit app layout
st.title("Sentiment Analysis of Info BMKG Reviews")

# Display sentiment distribution
st.subheader('Sentiment Distribution')
sns.countplot(x='sentiment', data=df_busu_clean)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
st.pyplot()

# Generate a word cloud
st.subheader('Word Cloud')
all_text = " ".join(df_busu_clean["text_stemmed"].dropna())
wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(all_text)
plt.figure(figsize=(5, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot()

# Display classification report
st.subheader('Model Evaluation')
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
st.subheader('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot()
