import sys
import io
import tkinter as tk
from tkinter import Label, messagebox
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
import os

# Use the directory of the current script (safe for deployment)
BACKEND_PATH = os.path.dirname(os.path.abspath(__file__))

# Redirect stdout to support UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def FacialExpressionModel(json_file, weights_file):
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        messagebox.showerror("Model Load Error", f"Failed to load model files.\n{e}")
        sys.exit()


# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Real-Time Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the face detection model
facec_path = os.path.join(BACKEND_PATH, 'haarcascade_frontalface_default.xml')
facec = cv2.CascadeClassifier(facec_path)

if facec.empty():
    messagebox.showerror("Haar Cascade Error", f"Failed to load Haar Cascade from:\n{facec_path}")
    print("Error loading Haar Cascade.")
    sys.exit()

# Load the facial expression model
model = FacialExpressionModel(
    os.path.join(BACKEND_PATH, "emotion_model.json"),
    os.path.join(BACKEND_PATH, "emotion_model.weights.h5")
)

# List of emotions
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


# Real-time emotion detection using webcam
def detect_emotion_in_real_time():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        label1.configure(foreground="#011638", text="Error accessing webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            try:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop and preprocess face
                face_crop = gray_image[y:y + h, x:x + w]
                roi = cv2.resize(face_crop, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)

                # Predict emotion
                pred = model.predict(roi, verbose=0)
                emotion_index = np.argmax(pred)
                emotion = EMOTIONS_LIST[emotion_index]

                # Display the emotion and prediction probabilities
                probabilities = ", ".join([f"{EMOTIONS_LIST[i]}: {pred[0][i]:.2f}" for i in range(len(EMOTIONS_LIST))])
                print(f"Predicted Emotion: {emotion}, Probabilities: {probabilities}")
                
                # Display the predicted emotion on frame
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error processing face: {str(e)}")

        # Display the frame
        cv2.imshow('Real-Time Emotion Detector', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()


def start_real_time_detection():
    label1.configure(foreground="#011638", text="Starting real-time detection...")
    detect_emotion_in_real_time()


# GUI Button to start detection
real_time_button = tk.Button(top, text="Start Real-Time Detection", command=start_real_time_detection, padx=10, pady=5)
real_time_button.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
real_time_button.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')

heading = Label(top, text='Real-Time Emotion Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
