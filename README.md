# 🧠 MoodMap – Backend (Flask API)

This is the backend of **MoodMap**, a real-time emotion detection app using webcam input.  
It uses **TensorFlow**, **OpenCV**, and a CNN-based facial emotion recognition model to detect human emotions from live video.

🔗 **Frontend Live Demo**: [https://super-snickerdoodle-46bef1.netlify.app](https://super-snickerdoodle-46bef1.netlify.app)

> 💻 The backend runs locally and serves real-time predictions through REST API and MJPEG video stream.

---

## 🔥 Features

- 🎥 Real-time emotion detection from webcam feed
- 😄 Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- ⚙️ REST API with `/start`, `/stop`, `/video_feed` endpoints
- 🧠 Powered by a trained deep learning model
- 🌐 Supports CORS for integration with frontend
- 🧼 Automatic cleanup on unmount

---

## 🧪 API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | `/start`          | Starts emotion detection & webcam |
| POST   | `/stop`           | Stops detection and releases camera |
| POST   | `/cleanup`        | Cleans up state (optional)        |
| GET    | `/video_feed`     | Returns live webcam MJPEG stream  |

---

## 🛠️ Tech Stack

- **Flask** – Python web framework
- **OpenCV** – Webcam and face detection
- **TensorFlow / Keras** – Emotion classification model
- **Gunicorn** – Production WSGI server
- **Flask-CORS** – To allow cross-origin requests from frontend

---

## ⚙️ Getting Started Locally

> ⚠️ Backend **must be run locally** due to webcam dependency.

### 1. Clone the repo

```bash
git clone https://github.com/anshul-3000/Moodmap_backend.git
cd Moodmap_backend/backend
```
## 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```
## 3. Install requirements
```bash
pip install -r requirements.txt
```
## 4. Ensure these files exist:
-app.py
-emotion_model.json
-emotion_model.weights.h5
-haarcascade_frontalface_default.xml

## 5. Run Flask app
```bash
python app.py
Runs on: http://localhost:5000
```

### 🚀 Deployment
- ✅ Backend not deployed on cloud due to webcam hardware access limitation.
- 🔄 All webcam detection is handled locally.
- 🌐 Frontend is hosted live and calls local backend.

### 🤝 Frontend Repository
[🔗 MoodMap Frontend GitHub](https://github.com/anshul-3000/Moodmap_frontend)
[🌍 Live Frontend on Netlify](https://super-snickerdoodle-46bef1.netlify.app/)

### 👨‍💻 Developed By
Anshul Chaudhary
B.Tech CSE | Full Stack + AI/ML Engineer
🚀 Passionate about AI-powered apps and innovative interfaces.
