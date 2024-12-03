import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import PorterStemmer
from google_play_scraper import Sort, reviews

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Streamlit App Title
st.title("Sentiment Analysis for Info BMKG Reviews")

# Input: App ID
app_id = st.text_input("Enter the App ID for analysis (default: com.Info_BMKG):", "com.Info_BMKG")

# Fetch reviews button
if st.button("Fetch Reviews"):
    # Fetch reviews
    result, _ = reviews(app_id, lang='id', country='id', sort=Sort.NEWEST, count=5000)
    df_busu = pd.DataFrame(np.array(result), columns=['review'])
    df_busu = df_busu.join(pd.DataFrame(df_busu.pop('review').tolist(), columns=['content', 'score', 'at', 'userName']))

    # Sentiment function
    def sentiment(score):
        if score <= 2:
            return 'Negatif'
        elif score == 3:
            return 'Netral'
        else:
            return 'Positif'

    # Apply sentiment function
    df_busu['sentiment'] = df_busu['score'].apply(sentiment)

    # Clean text function
    def clean_text(df, text_field, new_text_field_name):
        df[new_text_field_name] = df[text_field].str.lower()
        df[new_text_field_name] = df[new_text_field_name].apply(
            lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
        )
        df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
        return df

    # Apply cleaning
    df_busu_clean = clean_text(df_busu, 'content', 'text_clean')

    # Remove stopwords
    stop = stopwords.words('indonesian')
    df_busu_clean['text_StopWord'] = df_busu_clean['text_clean'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop])
    )

    # Apply stemming
    stemmer = PorterStemmer()
    df_busu_clean['text_stemmed'] = df_busu_clean['text_StopWord'].apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x.split()])
    )

    # Display DataFrame
    st.write("Sample Data:")
    st.write(df_busu_clean.head(10))

    # Visualize sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='sentiment', data=df_busu_clean, ax=ax)
    st.pyplot(fig)

    # Word Cloud
    st.subheader("Word Cloud")
    all_text = " ".join(df_busu_clean["text_stemmed"].dropna())
    wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(all_text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # Prepare data for machine learning
    X = df_busu_clean['text_stemmed']
    y = df_busu_clean['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train SVM model
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_tfidf, y_train)
    y_pred = svm_model.predict(X_test_tfidf)

    # Classification Report
    st.subheader("Model Evaluation")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negatif', 'Netral', 'Positif'], 
                yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
    st.pyplot(fig)

    # Save data to CSV
    st.download_button("Download Cleaned Data", data=df_busu_clean.to_csv(index=False), file_name="cleaned_data.csv")
