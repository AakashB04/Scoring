<<<<<<< HEAD
from flask import Flask, render_template, jsonify
=======
from flask import Flask, render_template, Response
>>>>>>> 848b39ecf55f7a1451b7f5be357364ffe8f1a66b
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
<<<<<<< HEAD
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
=======

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
>>>>>>> 848b39ecf55f7a1451b7f5be357364ffe8f1a66b
def get_face_score(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
<<<<<<< HEAD

        if emotion in ['happy', 'neutral']:
            return 0.9
        elif emotion == 'surprise':
            return 0.7
        else:
            return 0.4
    except Exception:
        return 0.5


=======
        if emotion in ['happy', 'neutral']:
            return 0.8
        elif emotion in ['surprise']:
            return 0.6
        else:
            return 0.3
    except:
        return 0.5

>>>>>>> 848b39ecf55f7a1451b7f5be357364ffe8f1a66b
def get_hand_score(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
<<<<<<< HEAD
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
=======
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
>>>>>>> 848b39ecf55f7a1451b7f5be357364ffe8f1a66b
@app.route('/')
def index():
    return render_template('index.html')

<<<<<<< HEAD

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
=======
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
>>>>>>> 848b39ecf55f7a1451b7f5be357364ffe8f1a66b
