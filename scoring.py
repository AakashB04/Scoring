import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from collections import deque

# -------------------------------
# Scoring parameters
w_f, w_h = 0.6, 0.4  # weights for face & hand
alpha = 0.2  # EMA smoothing factor
ema_prev = None


def update_ema(score):
    global ema_prev
    if ema_prev is None:
        ema_prev = score
    else:
        ema_prev = alpha * score + (1 - alpha) * ema_prev
    return max(0.0, min(1.0, ema_prev))  # keep between 0 and 1


# -------------------------------
# Mediapipe setup for hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# -------------------------------
# Helper: score facial expression
def get_face_score(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        if emotion in ['happy', 'neutral']:
            return 0.8  # good
        elif emotion in ['surprise']:
            return 0.6  # neutral
        else:
            return 0.3  # negative emotions
    except:
        return 0.5  # fallback neutral


# -------------------------------
# Helper: score hand gestures
def get_hand_score(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        score = 0.7  # natural hand presence
        for handLms in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        return score
    else:
        return 0.5  # no hands detected → neutral


# -------------------------------
# Main loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for webcam view

    # Get scores
    face_score = get_face_score(frame)
    hand_score = get_hand_score(frame)

    # Combine scores
    combined = w_f * face_score + w_h * hand_score

    # Smooth with EMA
    final_score = update_ema(combined)

    # Display info
    cv2.putText(frame, f"Face Score: {face_score:.2f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hand Score: {hand_score:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Final Non-Verbal Score: {final_score * 100:.1f}/100", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Interview Scoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
