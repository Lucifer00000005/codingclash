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

tags, patterns = [], []
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

# Utility Functions
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

def get_news(topic):
    url = "https://gnews.io/api/v4/search"
    params = {'q': topic, 'token': GNEWS_API_KEY, 'lang': 'en', 'max': 5}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if not articles:
            return f"Sorry, no latest news on {topic} right now."
        return "Here are the latest news headlines:\n" + "\n".join(
            [f"• {a['title']} ({a['source']['name']})" for a in articles]
        )
    return "Sorry, I couldn't fetch the news at the moment."

def predict_intent(text, threshold=0.3):
    if re.match(r'^[0-2],[0-2]$', text.strip()):
        return "tictactoe"
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    top_prob = max(probs)
    top_tag = model.classes_[probs.argmax()]
    print(f"[DEBUG] Input: '{text}' | Predicted: '{top_tag}' ({top_prob:.2f})")
    return top_tag if top_prob >= threshold else None

# --- Tic Tac Toe Game Logic ---
game_board = [[' ' for _ in range(3)] for _ in range(3)]
game_active = False

def print_board(board):
    return "\n".join(["  0 1 2"] + [f"{i} " + " ".join(row) for i, row in enumerate(board)])

def check_win(board, player):
    for i in range(3):
        if all([cell == player for cell in board[i]]) or all([board[r][i] == player for r in range(3)]):
            return True
    return all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)])

def check_draw(board):
    return all(cell != ' ' for row in board for cell in row)

def make_move(board, row, col, player):
    if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == ' ':
        board[row][col] = player
        return True
    return False

def minimax(board, depth, is_maximizing):
    if check_win(board, 'O'): return 1
    if check_win(board, 'X'): return -1
    if check_draw(board): return 0

    best = -float('inf') if is_maximizing else float('inf')
    for r in range(3):
        for c in range(3):
            if board[r][c] == ' ':
                board[r][c] = 'O' if is_maximizing else 'X'
                score = minimax(board, depth + 1, not is_maximizing)
                board[r][c] = ' '
                best = max(best, score) if is_maximizing else min(best, score)
    return best

def bot_move():
    best_score = -float('inf')
    best_move = None
    for r in range(3):
        for c in range(3):
            if game_board[r][c] == ' ':
                game_board[r][c] = 'O'
                score = minimax(game_board, 0, False)
                game_board[r][c] = ' '
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
    if best_move:
        game_board[best_move[0]][best_move[1]] = 'O'

def reset_game():
    global game_board, game_active
    game_board = [[' ' for _ in range(3)] for _ in range(3)]
    game_active = True

# --- Response Generator ---
def generate_response(intent, user_message):
    global game_active

    if intent == "weather":
        location = extract_location(user_message)
        return get_weather(location) if location else random.choice(responses_dict[intent])

    elif intent == "news":
        match = re.search(r'news (about|on|for) ([A-Za-z\s]+)', user_message, re.IGNORECASE)
        return get_news(match.group(2).strip()) if match else get_news('general')

    elif intent == "tictactoe":
        if re.search(r'start|play|new game|tic tac toe', user_message, re.IGNORECASE):
            reset_game()
            return "Game started! You are X. Send your move as row,col (e.g., 1,1).\n" + print_board(game_board)
        if not game_active:
            return "Please type 'start game' to begin."
        try:
            row, col = map(int, user_message.split(','))
            if make_move(game_board, row, col, 'X'):
                if check_win(game_board, 'X'):
                    game_active = False
                    return print_board(game_board) + "\nYou won! 🎉 Type 'start game' to play again."
                elif check_draw(game_board):
                    game_active = False
                    return print_board(game_board) + "\nIt's a draw. Type 'start game' to play again."
                else:
                    bot_move()
                    if check_win(game_board, 'O'):
                        game_active = False
                        return print_board(game_board) + "\nI won! 😎 Type 'start game' to play again."
                    elif check_draw(game_board):
                        game_active = False
                        return print_board(game_board) + "\nIt's a draw. Type 'start game' to play again."
                    else:
                        return print_board(game_board) + "\nYour move (row,col):"
            else:
                return "That spot is taken or invalid. Try again with row,col between 0 and 2."
        except:
            return "Invalid input. Please use the format row,col (e.g., 0,2)."

    elif intent in responses_dict:
        return random.choice(responses_dict[intent])

    return "I'm here to help. Please let me know how I can assist you."

# --- Flask Route ---
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
        print(f"[ERROR] {e}")
        return jsonify({"response": "Oops! Server error, please try again."})

if __name__ == '__main__':
    print("Flask app is running...")
    app.run(debug=True)
