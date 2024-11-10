from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from cvzone.HandTrackingModule import HandDetector
from openai import OpenAI
from gtts import gTTS
import os
import time
import threading
from queue import Queue, Empty
import pygame
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", host="127.0.0.1", port=5008)

# OpenAI client setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Text-to-speech setup
tts_queue = Queue()
pygame.mixer.init()

def tts_worker():
    print("TTS worker started")
    while True:
        try:
            print(f"Waiting for text. Queue size: {tts_queue.qsize()}")
            text = tts_queue.get(timeout=1)
            print(f"Got text from queue: {text}")
            if text is None:
                print("Received None, stopping worker")
                break
            tts = gTTS(text=text, lang='en')
            tts.save("temp.mp3")
            pygame.mixer.music.load("temp.mp3")
            print(f"Playing: {text}")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.remove("temp.mp3")
        except Empty:
            continue
        except Exception as e:
            print(f"Error in TTS worker: {e}")
        finally:
            tts_queue.task_done()
    print("TTS worker stopped")

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Define the model architecture and load weights
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(15, activation='softmax')
    ])
    model.load_weights('model5.weights.h5')
    return model

# Load the trained model and initialize hand detector
model = create_model()
detector = HandDetector(maxHands=2, detectionCon=0.8)

# Define class labels for predictions
label_class = {
    'class_0': 'Family', 'class_1': "How're you doing", 'class_2': "I'm Good", 'class_3': 'Goodbye',
    'class_4': 'Hello', 'class_5': 'Help', 'class_6': 'Party', 'class_7': 'No', 'class_8': 'Have a Great Day',
    'class_9': 'Sorry', 'class_10': 'Thank you', 'class_11': 'Yes', 'class_12': 'Wait', 'class_13': 'What',
    'class_14': 'Tonight'
}

# Initialize label tracking and timing
label_list = []
start_time = time.time()
last_detection_time = time.time()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/camera')
def page2():
    return render_template('page2.html')

@app.route('/login')
def login():
    return render_template('login.html')

def speak_text(text):
    print(f"Adding to queue: {text}")
    tts_queue.put(text)
    print(f"Queue size after adding: {tts_queue.qsize()}")

@socketio.on('image')
def handle_image(image_data):
    global label_list, start_time, last_detection_time

    try:
        # Decode the image
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Hand detection and processing
        try:
            hands, img = detector.findHands(frame, draw=False)
        except ValueError as e:
            print(f"Error in hand detection: {e}")
            hands = []

        if hands:
            # Get bounding box coordinates and crop the image
            x_min = min(hand['bbox'][0] for hand in hands)
            y_min = min(hand['bbox'][1] for hand in hands)
            x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
            y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)
            
            if x_min >= 0 and y_min >= 0 and x_max <= img.shape[1] and y_max <= img.shape[0]:
                img_crop = img[y_min:y_max, x_min:x_max]
                
                if img_crop.size != 0:
                    img_resize = cv2.resize(img_crop, (224, 224))
                    img_array = np.expand_dims(img_resize, axis=0)

                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[0][predicted_class]
                    print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

                    current_time = time.time()
                    if confidence > 0.7 and (current_time - last_detection_time) >= 1:
                        label = f"class_{predicted_class}"
                        prediction = label_class[label]
                        label_list.append(prediction)
                        emit('response', {'prediction': prediction, 'confidence': float(confidence)})
                        last_detection_time = current_time

        # Check if 6 seconds have passed and we have labels
        if time.time() - start_time > 6 and label_list:
            prompt = f"Make a sentence using only the labels provided here at max 5 words like a basic casual greeting: {', '.join(label_list)}"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are generating a sentence using labels we are providing."},
                    {"role": "user", "content": prompt}
                ]
            )

            reply = response.choices[0].message.content
            print(f"ChatGPT Response: {reply}")
            emit('chatgpt_response', {'response': reply})
            speak_text(reply)

            label_list.clear()
            start_time = time.time()

    except Exception as e:
        print(f"Error in handle_image: {e}")
        emit('error', {'message': str(e)})

@app.teardown_appcontext
def cleanup(exception):
    tts_queue.put(None)  # Signal the TTS thread to stop
    tts_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to stop
    pygame.mixer.quit()
    print("Cleaned up TTS resources.")

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5008)