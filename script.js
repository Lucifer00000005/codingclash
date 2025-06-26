let recognizing = false;
let recognition;
let ttsEnabled = false;

// Toggle speech recognition
function toggleSpeechRecognition() {
  if (!('webkitSpeechRecognition' in window)) {
    alert("Speech recognition not supported");
    return;
  }

  if (!recognition) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = function (event) {
      const transcript = event.results[0][0].transcript;
      document.getElementById('userInput').value = transcript;
      sendMessage();
    };

    recognition.onerror = function (event) {
      console.error("Speech recognition error", event);
    };

    recognition.onend = function () {
      recognizing = false;
    };
  }

  if (recognizing) {
    recognition.stop();
    recognizing = false;
  } else {
    recognition.start();
    recognizing = true;
  }
}

// Toggle Text-to-Speech (TTS)
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const btn = document.getElementById('ttsToggle');
  btn.textContent = ttsEnabled ? "🔊 TTS ON" : "🔇 TTS OFF";
  btn.style.backgroundColor = ttsEnabled ? "#128c7e" : "#25d366";
}

// Append chat messages
function appendMessage(message, sender) {
  const chatbox = document.getElementById('chatbox');
  const msg = document.createElement('div');
  msg.className = sender === "user" ? "user-message" : "bot-message";

  // Use <pre> for preformatted text like the board
  if (message.includes('+---') || message.includes('|')) {
    const pre = document.createElement('pre');
    pre.textContent = message;
    msg.appendChild(pre);
  } else {
    msg.textContent = message;
  }

  chatbox.appendChild(msg);
  chatbox.scrollTop = chatbox.scrollHeight;
}

// Send user message to the backend
function sendMessage() {
  const userInput = document.getElementById('userInput');
  const message = userInput.value.trim();
  if (message === "") return;

  appendMessage(message, "user");
  userInput.value = "";

  fetch('http://127.0.0.1:5000/chat', {
    method: 'POST',
    body: JSON.stringify({ message }),
    headers: { 'Content-Type': 'application/json' }
  })
    .then(response => response.json())
    .then(data => {
      const botReply = data.response;
      appendMessage(botReply, "bot");

      if (ttsEnabled) {
        const utterance = new SpeechSynthesisUtterance(botReply);
        speechSynthesis.speak(utterance);
      }
    })
    .catch(error => {
      console.error("Error:", error);
      appendMessage("Oops! Something went wrong.", "bot");
    });
}

// Event listeners
document.getElementById("userInput").addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    sendMessage();
  }
});

document.getElementById("ttsToggle").addEventListener("click", toggleTTS);
document.getElementById("micBtn").addEventListener("click", toggleSpeechRecognition);
