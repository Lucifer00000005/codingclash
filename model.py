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

# API Keys
WEATHER_API_KEY = '4b2d46632c267e5995c8dd1bbf9d6f84'
GNEWS_API_KEY = '0de65d25bb67f3c62a71b5e822a4d200'

# Utility functions
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

# Intent prediction
def predict_intent(text, threshold=0.3):
    if re.match(r'^[0-2],[0-2]$', text.strip()):
        return "tictactoe"

    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    top_prob = max(probs)
    top_tag = model.classes_[probs.argmax()]
    print(f"[DEBUG] Input: '{text}' | Predicted: '{top_tag}' ({top_prob:.2f})")
    if top_prob >= threshold:
        return top_tag
    return "fallback"

# Tic Tac Toe game logic
game_board = [[' ' for _ in range(3)] for _ in range(3)]
game_active = False

def print_board(board):
    display = "    0   1   2\n"
    display += "  +---+---+---+\n"
    for i, row in enumerate(board):
        display += f"{i} | " + " | ".join(row) + " |\n"
        display += "  +---+---+---+\n"
    return display

def check_win(board, player):
    for i in range(3):
        if all([cell == player for cell in board[i]]):
            return True
        if all([board[r][i] == player for r in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

def check_draw(board):
    return all(cell != ' ' for row in board for cell in row)

def make_move(board, row, col, player):
    if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':
        board[row][col] = player
        return True
    return False

def minimax(board, is_maximizing):
    winner_X = check_win(board, 'X')
    winner_O = check_win(board, 'O')
    if winner_X:
        return -1
    if winner_O:
        return 1
    if check_draw(board):
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'O'
                    score = minimax(board, False)
                    board[r][c] = ' '
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    board[r][c] = 'X'
                    score = minimax(board, True)
                    board[r][c] = ' '
                    best_score = min(score, best_score)
        return best_score

def bot_move():
    best_score = -float('inf')
    move = None
    for r in range(3):
        for c in range(3):
            if game_board[r][c] == ' ':
                game_board[r][c] = 'O'
                score = minimax(game_board, False)
                game_board[r][c] = ' '
                if score > best_score:
                    best_score = score
                    move = (r, c)
    if move:
        game_board[move[0]][move[1]] = 'O'

def reset_game():
    global game_board, game_active
    game_board = [[' ' for _ in range(3)] for _ in range(3)]
    game_active = True

# Generate responses
def generate_response(intent, user_message):
    global game_active

    if intent == "weather":
        location = extract_location(user_message)
        return get_weather(location) if location else random.choice(responses_dict[intent])

    elif intent == "news":
        topic_match = re.search(r'news (about|on|for) ([A-Za-z\s]+)', user_message, re.IGNORECASE)
        if topic_match:
            return get_news(topic_match.group(2).strip())
        return get_news("general")

    elif intent == "tictactoe":
        if re.search(r'start|play|new game|tic tac toe', user_message, re.IGNORECASE):
            reset_game()
            return "Game started! You are X. Send your move as row,col (e.g., 1,1).\n" + print_board(game_board)

        if not game_active:
            return "Game is not active. Type 'start game' to begin."

        move_match = re.match(r'^([0-2]),([0-2])$', user_message.strip())
        if move_match:
            row, col = int(move_match.group(1)), int(move_match.group(2))
            if make_move(game_board, row, col, 'X'):
                if check_win(game_board, 'X'):
                    game_active = False
                    return print_board(game_board) + "\nYou won! 🎉 To play again, type 'start game'."
                elif check_draw(game_board):
                    game_active = False
                    return print_board(game_board) + "\nIt's a draw! Type 'start game' to play again."
                else:
                    bot_move()
                    if check_win(game_board, 'O'):
                        game_active = False
                        return print_board(game_board) + "\nI won! 😎 Type 'start game' to play again."
                    elif check_draw(game_board):
                        game_active = False
                        return print_board(game_board) + "\nIt's a draw! Type 'start game' to play again."
                    else:
                        return print_board(game_board) + "\nYour turn! Send your move as row,col."
            else:
                return "Invalid move. Try again with an empty spot between 0,0 to 2,2."

        return "To play Tic Tac Toe, type 'start game' and then send your move like '1,2'."

    elif intent in responses_dict:
        return random.choice(responses_dict[intent])

    return random.choice(responses_dict["fallback"])

# Flask route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({"response": "Please send a valid message."})
        intent = predict_intent(user_input)
        response = generate_response(intent, user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"response": "Oops! Server error, please try again."})

if __name__ == '__main__':
    print("Flask app is running...")
    app.run(debug=True)
