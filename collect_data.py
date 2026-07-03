import cv2
import mediapipe as mp
import csv
import os

# ---------- SETUP ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

DATA_FILE = "isl_landmarks.csv"

# Create the CSV file with headers if it doesn't exist yet
if not os.path.exists(DATA_FILE):
    header = ["label"]
    for i in range(21):  # 21 hand landmarks
        header += [f"x{i}", f"y{i}", f"z{i}"]
    with open(DATA_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# ---------- WEBCAM LOOP ----------
cap = cv2.VideoCapture(0)

print("=" * 60)
print("ISL DATA COLLECTION")
print("Press a LETTER KEY (A-Z) to save the current hand pose as that label")
print("Press 'Q' to quit")
print("=" * 60)

sample_counts = {}

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access webcam.")
        break

    frame = cv2.flip(frame, 1)  # mirror view, feels natural
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    landmark_list = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten landmarks into a list: [x0,y0,z0, x1,y1,z1, ...]
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

    # Show instructions and live sample count on screen
    cv2.putText(frame, "Press A-Z to save sample | Q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    total_samples = sum(sample_counts.values())
    cv2.putText(frame, f"Total samples saved: {total_samples}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("ISL Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Check if key pressed is a letter A-Z
    if 97 <= key <= 122:  # lowercase a-z key codes
        letter = chr(key).upper()
        if landmark_list is not None:
            with open(DATA_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([letter] + landmark_list)
            sample_counts[letter] = sample_counts.get(letter, 0) + 1
            print(f"Saved sample for '{letter}' (total for this letter: {sample_counts[letter]})")
        else:
            print(f"No hand detected — sample for '{letter}' NOT saved. Try again.")

cap.release()
cv2.destroyAllWindows()

print("\nData collection finished.")
print("Samples collected per letter:", sample_counts)
print(f"Data saved to: {DATA_FILE}")
