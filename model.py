print("Starting the Flask server...")

from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import json
import random
import requests
import re

app = Flask(__name__)
CORS(app)

# Load intents
with open('intents.json', 'r') as f:
    data = json.load(f)

tags = []
patterns = []
responses_dict = {}

for intent in data['intents']:
    tag = intent['tag']
    responses_dict[tag] = intent['responses']
    for pattern in intent['patterns']:
        tags.append(tag)
        patterns.append(pattern)

df = pd.DataFrame({'pattern': patterns, 'tag': tags})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['pattern'])
y = df['tag']

model = SVC(probability=True)
model.fit(X, y)

# API KEYS
WEATHER_API_KEY = '4b2d46632c267e5995c8dd1bbf9d6f84'
GNEWS_API_KEY = '0de65d25bb67f3c62a71b5e822a4d200'

# Extract location from query
def extract_location(text):
    match = re.search(r'(?:in|for|at|on)\s+([A-Za-z\s,]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'([A-Za-z\s,]+)\s+weather', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if len(text.split()) <= 3:
        return text.strip()
    return None

# Weather
def get_weather(location):
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {'q': location, 'appid': WEATHER_API_KEY, 'units': 'metric'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        desc = data['weather'][0]['description'].capitalize()
        temp = data['main']['temp']
        city = data['name']
        return f"The weather in {city} is currently {desc} with a temperature of {temp}°C."
    return "Sorry, I couldn't find the weather for that location."

# News
def get_news(topic):
    url = "https://gnews.io/api/v4/search"
    params = {
        'q': topic,
        'token': GNEWS_API_KEY,
        'lang': 'en',
        'max': 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if not articles:
            return f"Sorry, no latest news on {topic} right now."
        return "Here are the latest news headlines:\n" + "\n".join(
            [f"• {a['title']} ({a['source']['name']})" for a in articles]
        )
    return "Sorry, I couldn't fetch the news at the moment."

# Predict intent
def predict_intent(text, threshold=0.3):
    try:
        if re.match(r'^[0-2],[0-2]$', text.strip()):
            return "tictactoe"

        vec = vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        top_prob = max(probs)
        top_tag = model.classes_[probs.argmax()]
        print(f"[DEBUG] Input: '{text}' | Predicted: '{top_tag}' ({top_prob:.2f})")

        if top_prob >= threshold:
            return top_tag
        return None
    except Exception as e:
        print(f"[ERROR] in predict_intent: {e}")
        return None

# Tic Tac Toe simplified version (you can replace this if you already have working game logic)
game_board = [[' ' for _ in range(3)] for _ in range(3)]
game_active = False

def print_board(board):
    board_display = "    0   1   2\n"
    board_display += "  +---+---+---+\n"
    for i, row in enumerate(board):
        board_display += f"{i} | " + " | ".join(row) + " |\n"
        board_display += "  +---+---+---+\n"
    return board_display

def reset_game():
    global game_board, game_active
    game_board = [[' ' for _ in range(3)] for _ in range(3)]
    game_active = True

# Response generator
def generate_response(intent, user_message):
    global game_active

    try:
        if intent == "weather":
            location = extract_location(user_message)
            return get_weather(location) if location else random.choice(responses_dict[intent])

        elif intent == "news":
            topic_match = re.search(r'news (about|on|for) ([A-Za-z\s]+)', user_message, re.IGNORECASE)
            if topic_match:
                return get_news(topic_match.group(2).strip())
            if re.search(r'news|latest news|headlines', user_message, re.IGNORECASE):
                return get_news('general')
            return random.choice(responses_dict[intent])

        elif intent == "tictactoe":
            if re.search(r'start|play|new game', user_message, re.IGNORECASE):
                reset_game()
                return "Game started! You are X. Send your move as row,col (e.g., 1,1).\n" + print_board(game_board)

            return "To play Tic Tac Toe, type 'start game' and then send moves like '0,1'."

        elif intent in responses_dict:
            return random.choice(responses_dict[intent])

    except Exception as e:
        print(f"[ERROR in generate_response] {e}")
        return "Something went wrong while processing your request."

    return "I'm here to help. Please let me know how I can assist you."

# Flask route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({"response": "Please send a valid message."})
        intent = predict_intent(user_input)
        response = generate_response(intent, user_input) if intent else "I'm not sure I understood. Could you please rephrase?"
        return jsonify({"response": response})
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return jsonify({"response": "Oops! Server error, please try again."})

if __name__ == '__main__':
    print("Flask app is running...")
    app.run(debug=True)
