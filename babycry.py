# -*- coding: utf-8 -*-
"""BabyCry.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IjcuxBvw6E_4sw_X76Gm67j1FFsx75OE
"""

import os
import pandas as pd
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
from collections import Counter

# Define raw audio dictionary
raw_audio = {}

# Loop through directories and label audio files
directories = ['hungry', 'belly_pain', 'burping', 'discomfort', 'tired']
for directory in directories:
    path = 'C:/Users/Gem/Documents/research streamlit/donateacry_corpus_cleaned_and_updated_data' + directory
    for filename in os.listdir(path):
        if filename.endswith(".wav"):
            raw_audio[os.path.join(path, filename)] = directory

# Define function to extract MFCC features and chop audio
def extract_mfcc(audio_file, max_length=100):
    audiofile, sr = librosa.load(audio_file)
    fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
    if fingerprint.shape[1] < max_length:
        pad_width = max_length - fingerprint.shape[1]
        fingerprint_padded = np.pad(fingerprint, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return fingerprint_padded.T
    elif fingerprint.shape[1] > max_length:
        return fingerprint[:, :max_length].T
    else:
        return fingerprint.T

# Chop audio and extract MFCC features for each track
X = []
y = []
max_length = 100
for i, (audio_file, label) in enumerate(raw_audio.items()):
    mfcc_features = extract_mfcc(audio_file, max_length=max_length)
    X.append(mfcc_features)
    y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Flatten the features and labels
X_flat = X.reshape(X.shape[0], -1)
y_flat = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_flat, test_size=0.2, random_state=42)

# Train and evaluate models
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=25, max_features=5)),
    ('Logistic Regression', LogisticRegression()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('SVM', SVC()),
]

print("Model, Accuracy, Precision, Recall")
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"{model_name}: {accuracy}, {precision}, {recall}")

print(X_train.shape)

# Reshape data for LSTM input
n_samples, n_features = X_train.shape[0], X_train.shape[1] // 100
n_timesteps = 100
X_train_lstm = X_train.reshape((n_samples, 100, 20))
n_samples_test = X_test.shape[0]
X_test_lstm = X_test.reshape((n_samples_test, n_timesteps, n_features))

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Define LSTM model
lstm_model = Sequential([
    LSTM(units=128, input_shape=(n_timesteps, n_features)),
    Dropout(0.2),
    Dense(units=64, activation='relu'),
    Dropout(0.2),
    Dense(units=len(np.unique(y_train_encoded)), activation='softmax')
])


# Compile LSTM model
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate LSTM model
_, accuracy = lstm_model.evaluate(X_test_lstm, y_test_encoded)
print("Accuracy:", accuracy)

import joblib

# Save the model
joblib.dump(lstm_model, "lstm_audio_model.joblib")

def pickle_model(model, modelname):
    directory = 'models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, str(modelname) + '.pkl'), 'wb') as f:
        return pickle.dump(model, f)

model = RandomForestClassifier()
model.fit(X_train, y_train)
pickle_model(model, "myRandomForest")

def getModel(pickle_path):
  with open(pickle_path, 'rb') as f:
        return pickle.load(f)

import joblib
model_path = "models/myRandomForest.pkl"  # Replace with your model path
with open(model_path, "rb") as f:
    model = joblib.load(f)

# pip install pydub

from io import BytesIO
from pydub import AudioSegment
import wave
import math
import os
import uuid

# Define the function to chop the audio
def chop_new_audio(audio_data, folder):
    os.makedirs(folder, exist_ok=True)  # Create directory if it doesn't exist
    audio = wave.open(audio_data, 'rb')
    frame_rate = audio.getframerate()
    n_frames = audio.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames / frame_rate))
    last_number_frames = 0

    for i in range(num_secs):
        shortfilename = str(uuid.uuid4())  # Generate a unique filename
        snippetfilename = f"{folder}/{shortfilename}snippet{i+1}.wav"
        snippet = wave.open(snippetfilename, 'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(audio.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.setnframes(audio.getnframes())
        snippet.writeframes(audio.readframes(window_size))
        audio.setpos(audio.tell() - 1 * frame_rate)

        # Check if the frame size of the snippet matches the previous snippets
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            os.rename(snippetfilename, f"{snippetfilename}.bak")
        snippet.close()

# Example usage
if __name__ == "__main__":
    # Example of reading audio data from a file-like object (e.g., uploaded file)
    with open('hungry baby.mp3', 'rb') as f:
        mp3_data = f.read()

    audio = AudioSegment.from_mp3(BytesIO(mp3_data))
    wav_data = BytesIO()
    audio.export(wav_data, format="wav")
    wav_data.seek(0)

    folder_name = "samples"
    chop_new_audio(wav_data, folder_name)
    print("Audio chopped successfully!")

# Predict on new audio snippets
predictions = []

folder_path = 'samples/'

for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        audiofile, sr = librosa.load(os.path.join(folder_path, filename))
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=20)
        fingerprint_flat = fingerprint.reshape(-1)  # Flatten the MFCC features
        # Pad or truncate features to match the number of features used for training
        if len(fingerprint_flat) < 2000:
            fingerprint_flat = np.pad(fingerprint_flat, (0, 2000 - len(fingerprint_flat)))
        elif len(fingerprint_flat) > 2000:
            fingerprint_flat = fingerprint_flat[:2000]
        prediction = model.predict([fingerprint_flat])  # Reshape to match expected input format
        predictions.append(prediction[0])

from collections import Counter
data = Counter(predictions)
print(data.most_common())  # Returns all unique items and their counts
print(data.most_common(1))  # Returns the most common prediction