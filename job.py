import pandas as pd

# Sample data
data = {
    'text': [
        "I love this product!",
        "This movie is amazing.",
        "I'm feeling happy today.",
        "I hate waiting in line.",
        "The weather is terrible.",
        "The service was slow and rude.",
        "The food tasted delicious.",
        "I'm so excited for the concert!",
        "I'm disappointed with the quality.",
        "The book was boring.",
    ],
    'sentiment': [
        'positive',
        'positive',
        'positive',
        'negative',
        'negative',
        'negative',
        'positive',
        'positive',
        'negative',
        'negative'
    ]
}
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import streamlit as st
# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('sample_data.csv', index=False)

# Preprocess data
X = df['text']
y = df['sentiment']

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train sentiment analysis model
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_tfidf, y)

# Save model
joblib.dump(svm_classifier, 'sentiment_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')

# Streamlit app
def predict_sentiment(text):
    # Load model and vectorizer
    model = joblib.load('sentiment_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    
    # Vectorize input text
    text_tfidf = vectorizer.transform([text])
    
    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]
    
    return prediction

def main():
    st.title("Sentiment Analysis App")
    
    # Text input for user to enter new text
    user_input = st.text_input("Enter text to analyze sentiment:")
    
    # Button to perform sentiment analysis
    if st.button("Analyze Sentiment"):
        if user_input:
            # Predict sentiment using the trained model
            prediction = predict_sentiment(user_input)
            
            # Display the sentiment prediction
            st.write(f"Predicted Sentiment: {prediction}")

if __name__ == "__main__":
    main()