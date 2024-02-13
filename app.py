
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request



# Load intents from JSON file
file_path = 'C:/Users/Randhir kumar/OneDrive/Desktop/chatbot project/intents.json'

try:
    with open(file_path, 'r') as file:
        intents_data = json.load(file)
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error decoding JSON. Check if the file is properly formatted.")

# Extract patterns and tags
patterns = []
tags = []
for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tags.append(tag)
        patterns.append(pattern)

# Tokenize patterns (convert text into a bag-of-words representation)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# Prepare labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Load the trained model
model = load_model('new_chatbot_model.h5')

def predict_tag(message):
    # Tokenize user input to match the format used for training
    user_input = vectorizer.transform([message])
    prediction = model.predict(user_input)
    predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_tag[0]

def get_response(predicted_tag):
    for intent in intents_data['intents']:
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            return np.random.choice(responses)
      

app = Flask(__name__)
app.static_folder = 'static'

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")  # This renders your HTML file

# Define the route to get responses from the bot
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    predicted_tag = predict_tag(userText)
    response = get_response(predicted_tag)
    return response

if __name__ == "__main__":
    app.run()