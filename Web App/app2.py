from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import re
import emoji
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load tokenizer, label encoder, vocabulary size, and stop words
with open('assets/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('assets/label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

with open('assets/vocabSize.pickle', 'rb') as handle:
    vocab_size = pickle.load(handle)

with open('assets/stopwords.pickle', 'rb') as handle:
    stop_words = pickle.load(handle)

# Load pre-trained model
model = load_model('assets/my_model91.h5')


def replace_emojis_with_text(text):
    if isinstance(text, str):
        return emoji.demojize(text)
    else:
        return ""


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@^_`{|}~"""), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalized_sentence(sentence):
    sentence = replace_emojis_with_text(sentence)  # Adding emoji replacement
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = removing_numbers(sentence)
    sentence = removing_punctuations(sentence)
    sentence = removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence


# Define a dictionary mapping numeric labels to words with emojis
emotion_labels = {
    0: 'Sadness 😔',
    1: 'Joy 😂',
    2: 'Fear 😱',
    3: 'Anger 😠',
    4: 'Love 🥰',
    5: 'Surprise 😲'
}

# Modify the predict_sentiment function to return emotion labels with emojis
def predict_sentiment(sentence):
    # Normalize the sentence
    sentence = normalized_sentence(sentence)

    # Tokenize the normalized sentence
    sentence = tokenizer.texts_to_sequences([sentence])

    # Pad the tokenized sentence
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

    # Predict using the model
    predictions = model.predict(sentence)
    predicted_emotion = np.argmax(predictions, axis=-1)[0]
    max_proba = np.max(predictions)
    
    # Get probabilities for all emotions
    probabilities = {emotion_labels[label]: proba for label, proba in enumerate(predictions[0])}
    
    return predicted_emotion, max_proba, probabilities

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/process_csv', methods=['POST'])
def process_csv():
    # Check if a CSV file is uploaded
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Check if the file is a CSV file
    if file and file.filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Iterate through rows and predict emotions
        emotions = []
        for index, row in df.iterrows():
            predicted_emotion, _, probabilities = predict_sentiment(row['text'])
            emotions.append(predicted_emotion)

        # Add 'label' column to the DataFrame
        df['Prediction'] = emotions
        
        # Save the DataFrame to a new CSV file
        new_filename = 'labeled_' + file.filename
        df.to_csv(new_filename, index=False)

        return jsonify({'message': 'Emotions labeled successfully', 'file': new_filename}), 200
    else:
        return jsonify({'message': 'Invalid file format. Please upload a CSV file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
