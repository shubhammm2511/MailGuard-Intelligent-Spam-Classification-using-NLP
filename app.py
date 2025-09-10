import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# --- Load Saved Objects (No change) ---
@st.cache_resource
def load_objects():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('word_features.pkl', 'rb') as f:
        word_features = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, word_features, encoder

model, word_features, encoder = load_objects()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


# --- Helper Functions (No change) ---
def preprocess_text(text):
    processed = re.sub(r'[^\w\d\s]', ' ', text)
    processed = re.sub(r'\s+', ' ', processed)
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    processed = processed.lower()
    words = word_tokenize(processed)
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [ps.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features


# --- Streamlit UI ---
st.set_page_config(page_title="MailGuard Spam Classifier", page_icon="🛡️", layout="wide")

# --- Initialize Session State for examples ---
if 'message' not in st.session_state:
    st.session_state.message = ""

def set_message(msg):
    st.session_state.message = msg

# --- Sidebar ---
with st.sidebar:
    st.title("🛡️ About MailGuard")
    st.info(
        "This app uses a Machine Learning model to classify messages as either legitimate (Ham) or Spam."
    )
    st.subheader("How It Works")
    st.markdown(
        """
        1.  Enter a message or use an example.
        2.  The app cleans the text.
        3.  The model predicts the category and its confidence.
        """
    )
    st.subheader("Model")
    st.markdown("The app uses an ensemble `VotingClassifier` with several underlying models for robust prediction.")


# --- Main Page ---
st.title("MailGuard: Intelligent Spam Classifier")
st.subheader("Is your message genuine or a potential threat? Let's find out.")

# --- Example Buttons ---
st.write("Try one of these examples:")
col1, col2 = st.columns(2)
with col1:
    st.button(
        "Example: Legitimate Message (Ham)",
        on_click=set_message,
        args=("Hey, are we still on for dinner at 8 PM tonight? Let me know!",),
        use_container_width=True
    )
with col2:
    st.button(
        "Example: Spam Message",
        on_click=set_message,
        args=("URGENT! You've won a $1000 prize. Click here to claim NOW http://bit.ly/claim-prize before it expires!",),
        use_container_width=True,
        type="secondary"
    )

# --- Input and Analysis ---
user_input = st.text_area("Your Message:", value=st.session_state.message, height=200, key="message_input")

if st.button("Analyze", use_container_width=True, type="primary"):
    if user_input:
        with st.spinner("🔍 Analyzing your message..."):
            # 1. Preprocess and Extract Features
            cleaned_input = preprocess_text(user_input)
            features = find_features(cleaned_input)

            # 2. Get Probability Distribution for prediction
            prob_dist = model.prob_classify(features)
            spam_confidence = prob_dist.prob(1)  # '1' corresponds to 'spam'
            ham_confidence = prob_dist.prob(0)   # '0' corresponds to 'ham'

            verdict = "SPAM" if spam_confidence > ham_confidence else "HAM"
            confidence = spam_confidence if verdict == "SPAM" else ham_confidence

            # 3. Display Results
            st.markdown("---")
            st.subheader("Analysis Result")
            
            col_verdict, col_confidence = st.columns(2)
            with col_verdict:
                if verdict == "SPAM":
                    st.metric("Verdict", "🚨 SPAM", "High Risk")
                else:
                    st.metric("Verdict", "✅ HAM", "Low Risk")

            with col_confidence:
                st.metric("Confidence Score", f"{confidence:.2%}")
            
            st.progress(confidence, text=f"Confidence: {confidence:.0%}")
            
            with st.expander("Show Processing Details"):
                st.write("**Original Message:**")
                st.info(user_input)
                st.write("**Cleaned Text (for model):**")
                st.warning(cleaned_input if cleaned_input else "No content after cleaning (e.g., only stopwords).")
    else:
        st.warning("Please enter a message to analyze.")