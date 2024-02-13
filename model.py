import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Load intents from JSON file
with open('C:/Users/Randhir kumar/OneDrive/Desktop/chatbot project/intents.json') as file:
    intents_data = json.load(file)

# Extract patterns and tags
patterns = []
tags = []
for intent in intents_data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        tags.append(tag)
        patterns.append(pattern)

# Tokenize patterns 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# Prepare labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Check sizes of X and y
print("X sizes:", X.shape[0])
print("y sizes:", len(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple neural network model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dense(64 , activation='relu'),
    Dense(len(set(y)), activation='softmax')
])
model.summary()
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X.toarray(), y, epochs=100, batch_size=8)

# Evaluate model on test set
y_pred_prob = model.predict(X_test.toarray())
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the class with highest probability as prediction

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
# Save the model for future use
model.save('new_chatbot_model.h5')
print("Model created and saved")


import tensorflow as tf

print("TensorFlow version:", tf.__version__)


