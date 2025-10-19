from flask import Flask, render_template, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import threading
import time

app = Flask(__name__)

# Global flags and score data
running = False
scores = []

# Setup Mediapipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# -------------------------------
# Helpers: Face and Hand Scoring
def get_face_score(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        if emotion in ['happy', 'neutral']:
            return 0.9
        elif emotion == 'surprise':
            return 0.7
        else:
            return 0.4
    except Exception:
        return 0.5


def get_hand_score(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        return 0.8
    else:
        return 0.5


# -------------------------------
# Background thread for scoring
def analyze_video():
    global running, scores
    cap = cv2.VideoCapture(0)
    scores = []

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        face_score = get_face_score(frame)
        hand_score = get_hand_score(frame)

        total_score = (0.6 * face_score) + (0.4 * hand_score)
        scores.append(total_score)

        cv2.putText(frame, f"Face: {face_score:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"Hand: {hand_score:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        cv2.putText(frame, f"Score: {total_score*100:.1f}/100", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("AI Interview Scoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start')
def start():
    global running
    if not running:
        running = True
        threading.Thread(target=analyze_video, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/stop')
def stop():
    global running
    running = False
    time.sleep(1)
    final_score = round(np.mean(scores), 2) if scores else 0
    return jsonify({'status': 'stopped', 'final_score': final_score})


if __name__ == '__main__':
    app.run(debug=True)
