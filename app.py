import os
import cv2
import numpy as np
from flask import Flask, Response, request
from tensorflow.keras.models import model_from_json
from tensorflow.keras import mixed_precision
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Enable mixed precision if needed (optional)
# mixed_precision.set_global_policy('mixed_float16')

# Load the face detection model
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion recognition model
def load_model(json_file, weights_file):
    with open(json_file, "r", encoding="utf-8") as file:
        model_json = file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load model
model = load_model("emotion_model.json", "emotion_model.weights.h5")

# Emotion labels
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Global camera state
camera = None
detecting = False

def generate_frames():
    global camera, detecting
    while detecting:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            try:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                pred = model.predict(roi)
                emotion = EMOTIONS_LIST[np.argmax(pred)]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print("Error processing face:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/start", methods=["POST"])
def start_detection():
    global camera, detecting
    if not detecting:
        camera = cv2.VideoCapture(0)
        detecting = True
    return "Detection started", 200

@app.route("/stop", methods=["POST"])
def stop_detection():
    global camera, detecting
    detecting = False
    if camera:
        camera.release()
        camera = None
    return "Detection stopped", 200

@app.route("/cleanup", methods=["POST"])
def cleanup_detection():
    return stop_detection()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
