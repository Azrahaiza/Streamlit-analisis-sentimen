# Install required libraries (uncomment if needed)
# !pip install google-play-scraper wordcloud seaborn nltk scikit-learn streamlit matplotlib tensorflow imbalanced-learn

# Import libraries
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from google_play_scraper import Sort, reviews
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import streamlit as st

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Fetch 5000 reviews for the app from Google Play Store
st.subheader("Mengambil 5000 Review dari Google Play Store")
app_id = "com.Info_BMKG"
result, _ = reviews(
    app_id,
    lang='id',
    country='id',
    sort=Sort.NEWEST,
    count=5000,
    filter_score_with=None
)

# Convert reviews to DataFrame
df_busu = pd.DataFrame(result)

# Define sentiment based on score
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
def clean_text(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9]+|(\w+:\/\/\S+)|^rt|http\S*|[^\w\s]", "", text)  # Remove unwanted characters
    text = re.sub(r"\d+", "", text)  # Remove numbers
    return text

df_busu['text_clean'] = df_busu['content'].apply(clean_text)

# Remove stopwords
stop = set(stopwords.words('indonesian'))
df_busu['text_StopWord'] = df_busu['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Apply stemming
stemmer = PorterStemmer()
df_busu['text_stemmed'] = df_busu['text_StopWord'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Display the first 5000 data
st.subheader("Menampilkan 5000 Data Review Awal")
st.dataframe(df_busu[['content', 'text_clean', 'sentiment']].head(5000))

# Display 10 data after stemming
st.subheader("10 Data Setelah Stemming")
st.dataframe(df_busu[['content', 'text_stemmed', 'sentiment']].head(10))

# Prepare data for machine learning
X = df_busu['text_stemmed']
y = df_busu['sentiment']

# Encode target labels
label_mapping = {'Negatif': 0, 'Netral': 1, 'Positif': 2}
y = y.map(label_mapping)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_tokenized, maxlen=100)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.3, random_state=42)

# Apply SMOTE for balancing the data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build LSTM model
embedding_dim = 128
model = Sequential([
    Embedding(input_dim=5000, output_dim=embedding_dim, input_length=100),
    LSTM(128, return_sequences=False),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # 3 classes: Negatif, Netral, Positif
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
st.subheader("Training Model LSTM...")
history = model.fit(X_train_resampled, y_train_resampled, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
st.subheader("Hasil Evaluasi Model LSTM")
loss, accuracy = model.evaluate(X_test, y_test)
st.text(f"Accuracy: {accuracy:.2f}")

# Predict on the test data
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Display Classification Report
st.subheader("Classification Report:")
st.text(classification_report(y_test, y_pred, target_names=['Negatif', 'Netral', 'Positif']))

# Plot Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Netral', 'Positif'], yticklabels=['Negatif', 'Netral', 'Positif'], ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
st.pyplot(fig)
