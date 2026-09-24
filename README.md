🛡️ MailGuard: Intelligent Spam Classifier
MailGuard is an intelligent web application designed to classify text messages (like SMS or emails) as either legitimate (ham) or spam. It leverages Natural Language Processing (NLP) and an ensemble Machine Learning model to provide fast and accurate real-time analysis.

🚀 Live Demo
https://mailguard-intelligent-spam-classification-using-nlp-hn7rtacxqr.streamlit.app/

✨ Features
Interactive Web Interface: A clean and engaging UI built with Streamlit for easy use.

Real-Time Prediction: Instantly analyzes user-submitted messages.

Confidence Scoring: Displays the model's confidence in its prediction for greater transparency.

Example Messages: Includes pre-written examples of both spam and ham for quick testing.

Ensemble ML Model: Powered by a robust Voting Classifier for high accuracy and reliable performance.

See the "Behind-the-Scenes": An expandable section shows how the input text is cleaned and processed before prediction.

⚙️ How It Works
The project follows a standard machine learning pipeline:

Text Preprocessing: When a user enters a message, the app first cleans the text. It converts it to lowercase, removes punctuation and common "stop words" (e.g., "the", "a", "is"), and reduces words to their root form (a process called stemming).

Feature Extraction: The cleaned text is converted into a numerical format using a Bag-of-Words technique. This checks for the presence of the 1,500 most common words from the training dataset.

Classification: The pre-trained ensemble model analyzes these features and classifies the message, providing a final verdict (Spam/Ham) and a confidence score.

🛠️ Technology Stack
Backend & ML: Python, Scikit-learn, NLTK, Pandas

Frontend: Streamlit

Model Training: Jupyter Notebook / Google Colab

Deployment: Streamlit Community Cloud

📂 Project Structure
spam-classifier/
│
├── 📜 app.py              # The main Streamlit application script
├── 📦 model.pkl            # The saved trained classifier model
├── 📝 word_features.pkl    # The saved list of word features
├── 🏷️ encoder.pkl         # The saved label encoder
├── 📄 requirements.txt    # Lists the required Python packages
└── 📄 README.md           # This file

📋 Setup and Installation
To run this project on your local machine, follow these steps:

1. Clone the Repository

git clone [https://github.com/your-username/spam-classifier.git](https://github.com/your-username/spam-classifier.git)
cd spam-classifier

2. Create a Virtual Environment (Recommended)

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

Note: The first time you run the app, NLTK may need to download necessary data packages (stopwords, punkt). The app script handles this.

4. Run the Streamlit App

streamlit run app.py

Your browser should automatically open to the app's local address.

📊 Dataset
The model was trained on the SMS Spam Collection Dataset from the UCI Machine Learning Repository, which contains over 5,500 labeled SMS messages.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
