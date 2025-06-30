import json
import random
import re
import logging
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import openai
import requests

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []
responses = {}
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern.lower())
        training_labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)
clf = SVC(probability=True)
clf.fit(X, training_labels)

# Flask app
app = Flask(__name__)

# API keys
openai.api_key = "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"
GNEWS_API_KEY = "0de65d25bb67f3c62a71b5e822a4d200"
OPENWEATHER_API_KEY = "4b2d46632c267e5995c8dd1bbf9d6f84"  # <-- Add your API key here

# Tic Tac Toe game state
game_state = {
    "board": [["" for _ in range(3)] for _ in range(3)],
    "user": "X",
    "bot": "O",
    "active": False
}

# --- Helper functions ---
def format_board():
    return "\n".join([" | ".join(cell if cell else " " for cell in row) for row in game_state["board"]])

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row): return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)): return True
    if all(board[i][i] == player for i in range(3)): return True
    if all(board[i][2 - i] == player for i in range(3)): return True
    return False

def is_full(board):
    return all(cell != "" for row in board for cell in row)

def minimax(board, is_maximizing):
    if check_winner(board, game_state["bot"]): return 1
    if check_winner(board, game_state["user"]): return -1
    if is_full(board): return 0

    if is_maximizing:
        best_score = -float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = game_state["bot"]
                    score = minimax(board, False)
                    board[i][j] = ""
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "":
                    board[i][j] = game_state["user"]
                    score = minimax(board, True)
                    board[i][j] = ""
                    best_score = min(score, best_score)
        return best_score

def best_move():
    best_score = -float("inf")
    move = None
    for i in range(3):
        for j in range(3):
            if game_state["board"][i][j] == "":
                game_state["board"][i][j] = game_state["bot"]
                score = minimax(game_state["board"], False)
                game_state["board"][i][j] = ""
                if score > best_score:
                    best_score = score
                    move = (i, j)
    return move

# --- Flask route ---
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    logging.debug(f"User input: {user_input}")

    # Handle Tic Tac Toe move
    if re.match(r"^\d,\d$", user_input):
        if not game_state["active"]:
            return jsonify({"response": "Game not active. Type 'tictactoe' to start."})
        i, j = map(int, user_input.split(","))
        if not (0 <= i <= 2 and 0 <= j <= 2):
            return jsonify({"response": "Invalid cell. Use format row,col (0–2)."})
        if game_state["board"][i][j] != "":
            return jsonify({"response": "Cell already taken. Try another."})

        game_state["board"][i][j] = game_state["user"]
        if check_winner(game_state["board"], game_state["user"]):
            game_state["active"] = False
            return jsonify({"response": f"You win!\n{format_board()}"})
        if is_full(game_state["board"]):
            game_state["active"] = False
            return jsonify({"response": f"It's a draw!\n{format_board()}"})

        move = best_move()
        if move:
            game_state["board"][move[0]][move[1]] = game_state["bot"]
        if check_winner(game_state["board"], game_state["bot"]):
            game_state["active"] = False
            return jsonify({"response": f"I win!\n{format_board()}"})
        if is_full(game_state["board"]):
            game_state["active"] = False
            return jsonify({"response": f"It's a draw!\n{format_board()}"})
        return jsonify({"response": f"Your move:\n{format_board()}"})

    # Start Tic Tac Toe
    if "tictactoe" in user_input:
        game_state["board"] = [["" for _ in range(3)] for _ in range(3)]
        game_state["active"] = True
        return jsonify({"response": "Game started! You are X. Use row,col (0–2).\n" + format_board()})

    # Weather handler
    if "weather" in user_input:
        city = user_input.replace("weather", "").replace("in", "").strip()
        if not city:
            return jsonify({"response": "Please specify a city, like 'weather in Delhi'."})
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
            res = requests.get(url).json()
            if res.get("cod") != 200:
                return jsonify({"response": f"Couldn't fetch weather for {city.title()}."})
            temp = res["main"]["temp"]
            desc = res["weather"][0]["description"]
            return jsonify({"response": f"Weather in {city.title()}: {desc}, {temp}°C"})
        except Exception as e:
            logging.error(f"Weather error: {e}")
            return jsonify({"response": "Weather fetch failed."})

    # News handling
    if "news" in user_input:
        topic = user_input.replace("news", "").strip() or "general"
        url = f"https://gnews.io/api/v4/search?q={topic}&lang=en&token={GNEWS_API_KEY}"
        try:
            res = requests.get(url).json()
            articles = res.get("articles", [])
            if not articles:
                return jsonify({"response": f"No news found for {topic}."})
            headlines = [f"- {article['title']}" for article in articles[:5]]
            return jsonify({"response": f"Top {topic} news:\n" + "\n".join(headlines)})
        except Exception as e:
            logging.error(f"News error: {e}")
            return jsonify({"response": "News fetch failed. Try again later."})

    # Code generation
    if "generate code" in user_input or "write code" in user_input:
        lang = "python" if "python" in user_input else "c++" if "c++" in user_input else "python"
        prompt = f"Generate {lang} code for: {user_input}"
        try:
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return jsonify({"response": completion.choices[0].message.content.strip()})
        except Exception as e:
            logging.error(f"Codegen error: {e}")
            return jsonify({"response": "Code generation failed. Try again later."})
    # Intent classification fallback
    try:
        vec = vectorizer.transform([user_input])
        prediction = clf.predict(vec)[0]
        logging.debug(f"Predicted intent: {prediction}")
        return jsonify({"response": random.choice(responses.get(prediction, responses["fallback"]))})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"response": random.choice(responses.get("fallback", ["Sorry, I didn't understand that."]))})

# Serve frontend
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# Start server
if __name__ == "__main__":
    app.run(debug=True)
