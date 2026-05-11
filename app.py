import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load model and vectorizer
tfidf = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

# Streamlit UI
st.title("Spam Shield SMS Sentinel")

input_sms = st.text_area("Enter the message", key="sms_input")

if st.button('Predict'):

    # Check empty input
    if input_sms.strip() == "":
        st.warning("No text found. Please enter a message.")

    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.error("Spam Message")
        else:
            st.success("Not Spam Message")