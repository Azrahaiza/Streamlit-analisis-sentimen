# Install required libraries
#pip install imbalanced-learn==0.10.1
#pip install tensorflow==2.13.0
#pip install scikit-learn==1.2.2
#pip install pandas==1.5.3
# pip install nltk==3.8
#pip install matplotlib==3.6.3
#pip install seaborn==0.12.2
#pip install wordcloud==1.8.2
#pip install streamlit==1.24.0

# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the app ID for Info BMKG
app_id = "com.Info_BMKG"

# Fetch reviews for the app from Google Play Store
# For testing, we can load a smaller subset or mock data
df_busu = pd.read_csv('data_reviews_with_sentiment.csv')  # You can replace this with your actual dataset

# Sentiment function based on the score
def sentiment(score):
    if score <= 2:
        return 'Negatif'
    elif score == 3:
        return 'Netral'
    else:
        return 'Positif'

# Clean the text data
def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()  # Convert to lowercase
    df[new_text_field_name] = df[new_text_field_name].apply(
        lambda elem: re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", elem)
    )
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

# Display the first 10 rows of the stemmed data
st.write("First 10 stemmed reviews:")
st.write(df_busu_clean.head(10))

# Visualize the sentiment distribution
sns.countplot(x='sentiment', data=df_busu_clean)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
st.pyplot(plt)

# Prepare data for machine learning (TF-IDF Vectorization)
X = df_busu_clean['text_stemmed']
y = df_busu_clean['sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

# Train a Support Vector Classifier (SVC) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_smote, y_train_smote)

# Predict on the test data
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate the model
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
st.pyplot(plt)

# Generate a word cloud
all_text = " ".join(df_busu_clean["text_stemmed"].dropna())
wordcloud = WordCloud(width=700, height=400, background_color="white", colormap="jet").generate(all_text)

# Display the word cloud
plt.figure(figsize=(5, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)

# Save the data with sentiment to CSV
df_busu_clean.to_csv('data_reviews_with_sentiment.csv', index=False)

# Streamlit file download button
st.download_button(
    label="Download cleaned data with sentiment",
    data=df_busu_clean.to_csv(index=False),
    file_name="data_reviews_with_sentiment.csv",
    mime="text/csv"
)
