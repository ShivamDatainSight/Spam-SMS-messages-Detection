import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


'''import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Initialize stemmer and stopwords once
ps = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def transform_text(text):
    text = text.lower()

    # Tokenization
    words = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    words = [word for word in words if word.isalnum()]

    # Remove stopwords and punctuation
    words = [
        word for word in words
        if word not in STOP_WORDS and word not in string.punctuation
    ]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)


@st.cache_resource
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


# Load model and vectorizer
tfidf, model = load_models()

# App UI
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email/SMS Spam Classifier")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

input_sms = st.text_area(
    "Message",
    placeholder="Type or paste your message here..."
)

if st.button("Predict", use_container_width=True):

    if not input_sms.strip():
        st.warning("Please enter a message.")
    else:
        # Preprocess text
        transformed_sms = transform_text(input_sms)

        # Convert to vector
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam Message")

        # Show confidence if supported
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(vector_input)[0]
            confidence = max(probability)

            st.write(f"**Confidence:** {confidence:.2%}")
            st.progress(float(confidence))'''
