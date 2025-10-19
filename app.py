from flask import Flask, render_template, Response
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp

# ------------------------------
# Flask setup
app = Flask(__name__)

# ------------------------------
# Scoring parameters
w_f, w_h = 0.6, 0.4
alpha = 0.2
ema_prev = None

def update_ema(score):
    global ema_prev
    if ema_prev is None:
        ema_prev = score
    else:
        ema_prev = alpha * score + (1 - alpha) * ema_prev
    return max(0.0, min(1.0, ema_prev))

# ------------------------------
# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ------------------------------
# Scoring functions
def get_face_score(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        if emotion in ['happy', 'neutral']:
            return 0.8
        elif emotion in ['surprise']:
            return 0.6
        else:
            return 0.3
    except:
        return 0.5

def get_hand_score(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        return 0.7
    else:
        return 0.5

# ------------------------------
# Video generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    global ema_prev

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        face_score = get_face_score(frame)
        hand_score = get_hand_score(frame)

        combined = w_f * face_score + w_h * hand_score
        final_score = update_ema(combined)

        cv2.putText(frame, f"Face: {face_score:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Hand: {hand_score:.2f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Score: {final_score*100:.1f}/100", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ------------------------------
# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
