from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import joblib
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load models
try:
    lstm_model = load_model('lstm_model.h5')
    nb_model = joblib.load('nb_pipeline.pkl')
    tokenizer = joblib.load('tokenizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")

# Preprocessing function (for LSTM model)
stop_words = set(['the', 'and', 'is', 'in', 'it', 'this', 'to'])  # Add stop words as per your list
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stop_words, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = [token for token in text.split() if token not in stop_words]
    return " ".join(tokens)

# Max sequence length
MAX_SEQUENCE_LENGTH = 30

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the POST request
        text = request.form.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the text
        cleaned_text = preprocess(text, stop_words)
        
        # Option to use Naive Bayes or LSTM
        model_choice = request.form.get('model')
        
        if model_choice == 'lstm':
            # Tokenize and pad text for LSTM model
            seq = tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
            
            # Predict sentiment using LSTM model
            pred = lstm_model.predict(padded)[0][0]
            sentiment = "Positive" if pred > 0.5 else "Negative"
        
        elif model_choice == 'nb':
            # Predict sentiment using Naive Bayes model
            sentiment = nb_model.predict([cleaned_text])[0]
        
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
        
        return jsonify({'sentiment': sentiment})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
