import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time

# ---------- LOAD CUSTOM MODEL ----------
print("Loading your custom model...")
model = joblib.load("isl_model_custom.pkl")
print("Model loaded. Classes it knows:", list(model.classes_))

# ---------- TEXT-TO-SPEECH ----------
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("=" * 60)
print("ISL REAL-TIME APP (custom model)")
print("Hold a sign steady to add its letter to the sentence")
print("Press 'C' to clear text | 'S' to speak the sentence | 'Q' to quit")
print("=" * 60)

current_text = ""
last_letter = None
letter_start_time = None
HOLD_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.6

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    predicted_letter = None
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            X_input = np.array(landmark_list).reshape(1, -1)
            probabilities = model.predict_proba(X_input)[0]
            best_idx = np.argmax(probabilities)
            confidence = probabilities[best_idx]
            predicted_letter = model.classes_[best_idx]

    # ---------- HOLD-TO-CONFIRM LOGIC ----------
    if predicted_letter is not None and confidence >= CONFIDENCE_THRESHOLD:
        if predicted_letter == last_letter:
            elapsed = time.time() - letter_start_time
            if elapsed >= HOLD_DURATION:
                current_text += predicted_letter
                print(f"Added letter: {predicted_letter} -> Current text: {current_text}")
                last_letter = None
                letter_start_time = None
        else:
            last_letter = predicted_letter
            letter_start_time = time.time()
    else:
        last_letter = None
        letter_start_time = None

    # ---------- DISPLAY ----------
    if predicted_letter is not None:
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.putText(frame, f"Predicted: {predicted_letter} ({confidence*100:.0f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if last_letter == predicted_letter and letter_start_time is not None:
            elapsed = time.time() - letter_start_time
            progress = min(elapsed / HOLD_DURATION, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (10, 40), (10 + 200, 55), (100, 100, 100), 2)
            cv2.rectangle(frame, (10, 40), (10 + bar_width, 55), (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, f"Text: {current_text}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, "C: clear | S: speak | Q: quit", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ISL Real-Time App", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_text = ""
        print("Text cleared.")
    elif key == ord('s'):
        if current_text:
            print(f"Speaking: {current_text}")
            speak(current_text)
        else:
            print("No text to speak yet.")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal text: {current_text}")
