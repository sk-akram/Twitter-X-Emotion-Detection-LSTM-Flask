from flask import Flask, render_template, request
import pandas as pd
from ntscraper import Nitter
import pickle
import re
import emoji
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
from flask import jsonify
from wordcloud import WordCloud
import base64


scraper1 = Nitter()

app = Flask(__name__)




# Load tokenizer, label encoder, vocabulary size, and stop words
with open(r'assets\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open(r'assets\label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

with open(r'assets\vocabSize.pickle', 'rb') as handle:
    vocab_size = pickle.load(handle)

with open(r'assets\stopwords.pickle', 'rb') as handle:
    stop_words = pickle.load(handle)

# Load pre-trained model
model = load_model(r'assets\my_model91.h5')


def replace_emojis_with_text(text):
    if isinstance(text, float):
        return str(text)  # Convert float to string
    return emoji.demojize(text)


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


def predict_sentiment(sentence):
    # Normalize the sentence
    sentence = normalized_sentence(sentence)

    # Tokenize the normalized sentence
    sentence = tokenizer.texts_to_sequences([sentence])

    # Pad the tokenized sentence
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

    # Predict using the model
    predictions = model.predict(sentence)
    predicted_emotion = le.inverse_transform(np.argmax(predictions, axis=-1))[0]
    max_proba = np.max(predictions)
    
    # Get probabilities for all emotions
    probabilities = {emotion: proba for emotion, proba in zip(le.classes_, predictions[0])}
    
    return predicted_emotion, max_proba, probabilities




# Function to fetch user information
def get_user_info(username):
    global scraper1
    user_info = scraper1.get_profile_info(username=username[1:])
    return user_info

def format_number(value):
    if value < 1000:
        return str(value)
    elif value < 1000000:
        return '{:.1f}K'.format(value / 1000)
    elif value < 1000000000:
        return '{:.1f}M'.format(value / 1000000)
    else:
        return '{:.1f}B'.format(value / 1000000000)

# Register the custom filter in the Flask application
app.jinja_env.filters['format_number'] = format_number


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    username = request.form['username']
    if username.startswith('#'):
        data = {
            "Name": username[1:].capitalize(),
            "Username": username,
            "Followers": 0,
            "Following": 0,
            "user_pic": r'https://t3.ftcdn.net/jpg/02/66/49/72/360_F_266497240_sbKnQ0BEoOPafo9JaefZZz00WK7t8LHq.jpg'
        }
    else:
        user_info = get_user_info(username)
        data = {
            "Name": user_info['name'],
            "Username": user_info['username'],
            "Followers": user_info['stats']['followers'],
            "Following": user_info['stats']['following'],
            "user_pic": user_info['image']
        }
    return render_template('result.html', data=data)


from collections import defaultdict

# combined_text = ' '.join(posts)
#     combined_text = normalized_sentence(combined_text)
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

@app.route('/detect_emotions', methods=['POST'])
def detect_emotions():
    global scraper1
    username = request.form['username']
    if username.startswith('@'):
        tweets = scraper1.get_tweets(username[1:], mode='user', number=100)
        posts = [tweet['text'] for tweet in tweets['tweets']]

    elif username.startswith('#'):
        tweets = scraper1.get_tweets(username[1:], mode='hashtag', number=100)
        # print(tweets,'//////////////////////////////////////////////////')
        posts = [tweet['text'] for tweet in tweets['tweets']]
        
    dict2 = {'0':0, '1':0, '2':0, '3':0, '4':0, '5':0}
    combined_text = normalized_sentence(' '.join(posts))
    
    # Generate word cloud
    wordcloud = WordCloud(width=400, height=400, background_color='white').generate(combined_text)
    # Save word cloud as image
    wordcloud_path = r'assets\wordcloud.png'
    wordcloud.to_file(wordcloud_path)

    with open(wordcloud_path, 'rb') as f:
        wordcloud_base64 = base64.b64encode(f.read()).decode('utf-8')

    for sentence in posts:
        predicted_emotion, _, probabilities = predict_sentiment(sentence)
        for d in dict2.keys():
            dict2[d] += probabilities[int(d)]
            
    total_sum = sum(dict2.values())

    # Calculate percentages
    percentages_dict = {int(k): (v / total_sum * 100) for k, v in dict2.items()}
    
    percentage_dict = dict(sorted(percentages_dict.items(), key=lambda item: item[1], reverse=True))
    print(percentage_dict)
    response_data = {
        "wordcloud": wordcloud_base64,
        "emotion_percentages": percentage_dict
    }
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
