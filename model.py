import json
import random
import re
import logging
import requests
from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from bs4 import BeautifulSoup
from googlesearch import search

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Train model
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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)
clf = SVC(probability=True)
clf.fit(X, training_labels)

# Flask app
app = Flask(__name__)

# GNews API Key
GNEWS_API_KEY = "0de65d25bb67f3c62a71b5e822a4d200"

# Game state for Tic Tac Toe
game_state = {
    "board": [["" for _ in range(3)] for _ in range(3)],
    "user": "X",
    "bot": "O",
    "active": False
}

# ---------- Tic Tac Toe Utilities ----------
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

# ---------- External Feature Functions ----------
def fetch_code_from_gfg(query):
    try:
        logging.debug("Searching GFG for: " + query)
        urls = list(search(f"site:geeksforgeeks.org {query}", num_results=5))
        gfg_url = next((url for url in urls if "geeksforgeeks.org" in url), None)
        if not gfg_url:
            return "No article found on GFG."
        res = requests.get(gfg_url)
        soup = BeautifulSoup(res.text, "html.parser")
        code_blocks = soup.find_all("pre")
        if not code_blocks:
            return "No code found in the GFG article."
        return code_blocks[0].text.strip()
    except Exception as e:
        logging.error(f"GFG scraping error: {e}")
        return "Failed to fetch code from GFG."

def scrape_travel_places(location):
    try:
        logging.debug(f"Searching Holidify for: {location}")
        query = f"{location} top places to visit site:holidify.com"
        urls = list(search(query, num_results=5))
        url = next((u for u in urls if "holidify.com" in u), None)
        if not url:
            return "Sorry, I couldn't find travel info for that location."
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        titles = soup.find_all("h3")
        places = [title.text.strip() for title in titles if len(title.text.strip()) > 3][:5]
        if not places:
            return "Couldn't find tourist attractions for that location."
        return f"Top places to visit in {location.title()}:\n- " + "\n- ".join(places)
    except Exception as e:
        logging.error(f"Trip planner scrape error: {e}")
        return "Something went wrong while fetching travel recommendations."

def get_fitness_tips():
    try:
        url = "https://www.healthline.com/health/fitness-exercise"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        tips = soup.find_all("a", class_="css-1egxyvc")
        clean_tips = []
        for tip in tips[:6]:
            text = tip.get_text(strip=True)
            if 10 < len(text) < 100:
                clean_tips.append(f"• {text}")
        return "\n".join(["Here are some fitness tips:"] + clean_tips) if clean_tips else None
    except Exception as e:
        logging.error(f"Fitness scraping error: {e}")
        return None

def get_makeup_tips(user_input=""):
    user_input = user_input.lower()

    if "oily" in user_input:
        return "💧 Oily Skin Tips:\n• Use matte, oil-free foundation.\n• Blotting papers help remove shine.\n• Avoid heavy creams.\n• Set makeup with powder."

    if "dry" in user_input:
        return "🌿 Dry Skin Tips:\n• Use hydrating primer & cream foundation.\n• Avoid powders.\n• Blend with damp sponge.\n• Finish with dewy mist."

    if "acne" in user_input or "acne-prone" in user_input:
        return "🌼 Acne-Prone Skin Tips:\n• Use non-comedogenic foundation.\n• Avoid clogging heavy concealers.\n• Clean tools regularly.\n• Don’t skip moisturizer."

    if "combination" in user_input:
        return "🌟 Combination Skin Tips:\n• Matte foundation on T-zone.\n• Hydrating on dry areas.\n• Use semi-matte blends.\n• Customize per zone."

    if "foundation" in user_input or "blend" in user_input:
        return "🧴 Foundation Guide:\n• Oily: Matte, oil-free.\n• Dry: Dewy, cream.\n• Acne: Non-comedogenic.\n• Combo: Semi-matte."

    try:
        url = "https://www.byrdie.com/makeup-tips-4847098"
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        tips = [p.text.strip() for p in paragraphs if 30 < len(p.text.strip()) < 250]
        if tips:
            return "💄 General Makeup Tips:\n" + "\n".join([f"• {tip}" for tip in tips[:5]])
    except:
        pass

    fallback_tips = [
        "Always prep your skin with a moisturizer.",
        "Use primer to extend makeup longevity.",
        "Choose foundation that matches your undertone.",
        "Use setting spray for longer wear.",
        "Remove makeup before bed to avoid breakouts."
    ]
    return "💄 General Makeup Tips:\n" + "\n".join([f"• {tip}" for tip in fallback_tips])

dating_tips = [
    "• Be yourself. Authenticity is attractive.",
    "• Listen more than you speak.",
    "• Keep your first date light and casual.",
    "• Practice good hygiene and grooming.",
    "• Stay off your phone and give your full attention.",
]

# ---------- Chat Route ----------
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip().lower()
    logging.debug(f"User input: {user_input}")

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

    if "tictactoe" in user_input:
        game_state["board"] = [["" for _ in range(3)] for _ in range(3)]
        game_state["active"] = True
        return jsonify({"response": "Game started! You are X. Use row,col (0–2).\n" + format_board()})

    if "weather" in user_input:
        city = user_input.replace("weather", "").strip()
        if not city:
            return jsonify({"response": "Please specify a city name."})
        try:
            res = requests.get(f"https://wttr.in/{city}?format=3")
            return jsonify({"response": res.text if res.status_code == 200 else "Couldn't fetch weather info."})
        except Exception as e:
            logging.error(f"Weather error: {e}")
            return jsonify({"response": "Error fetching weather."})

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

    if "generate code" in user_input or "write code" in user_input:
        prompt = user_input.replace("generate code", "").replace("write code", "").strip()
        if not prompt:
            return jsonify({"response": "Please specify the code you want me to generate."})
        return jsonify({"response": fetch_code_from_gfg(prompt)})

    if any(word in user_input for word in ["trip", "vacation", "places to visit", "travel"]):
        match = re.search(r"(trip to|visit|in|to)\s+([a-zA-Z\s]+)", user_input)
        location = match.group(2).strip() if match else None
        if not location:
            return jsonify({"response": "Please mention a location to get travel suggestions."})
        return jsonify({"response": scrape_travel_places(location)})

    if "fitness" in user_input or "health tips" in user_input:
        tips = get_fitness_tips()
        return jsonify({"response": tips if tips else "Sorry, I couldn't fetch fitness tips right now."})

    if any(word in user_input for word in ["makeup", "foundation", "blend", "skin"]):
        return jsonify({"response": get_makeup_tips(user_input)})

    if "dating" in user_input:
        return jsonify({"response": "Here are some dating tips:\n" + "\n".join(dating_tips)})

    try:
        vec = vectorizer.transform([user_input])
        prediction = clf.predict(vec)[0]
        logging.debug(f"Predicted intent: {prediction}")
        return jsonify({"response": random.choice(responses.get(prediction, responses["fallback"]))})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"response": random.choice(responses.get("fallback", ["Sorry, I didn't understand that."]))})

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

if __name__ == "__main__":
    app.run(debug=True)
 
