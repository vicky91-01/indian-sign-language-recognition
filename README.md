# Real-Time Indian Sign Language Recognition System

A real-time system that recognizes Indian Sign Language (ISL) hand signs through a webcam and converts them into text and speech — built to explore accessibility technology for the deaf and hard-of-hearing community. (ISL is the sign language used by the deaf community in India, similar in purpose to American Sign Language / ASL, but visually distinct.)

## What it does

The application uses a webcam to detect hand signs in real time, predicts the corresponding letter using a trained machine learning model, builds up a sentence as the user signs multiple letters, and can speak the resulting text aloud.

**Currently supports:** A, B, C, D, K, L, O, V, W, Y , I (11 letters)

## Why this project

Indian Sign Language has very few dedicated recognition tools compared to American Sign Language (ASL), despite India having millions of deaf and hard-of-hearing individuals with limited access to interpreters. This project explores a lightweight, low-cost, camera-only approach that could eventually support everyday communication in situations where an interpreter isn't available.

## How it works

1. **Hand tracking:** [MediaPipe](https://google.github.io/mediapipe/) detects the hand in each webcam frame and extracts 21 landmark points (x, y, z coordinates for knuckles, fingertips, and palm).
2. **Feature extraction:** These 21 points are flattened into a 63-number feature vector representing the hand's shape and position.
3. **Classification:** A Random Forest classifier (scikit-learn), trained on self-collected labeled samples, predicts which letter the hand shape represents.
4. **Confirmation logic:** A sign must be held steady for 1.5 seconds before its letter is added to the output text, to avoid accidental/flickering predictions.
5. **Speech output:** The built sentence can be converted to speech using `pyttsx3` (offline text-to-speech).

## Model performance

Trained on 890 self-recorded samples across 10 letters.

- **Test accuracy: 93.82%**
- Per-letter performance ranges from 0.84–1.00 F1-score
- Full classification report available in the training logs

## Tech stack

- Python 3.10
- MediaPipe (hand landmark detection)
- OpenCV (webcam capture and display)
- scikit-learn (Random Forest classifier)
- pyttsx3 (offline text-to-speech)
- pandas / NumPy (data handling)

## Project structure

```
ISL-Translator/
├── collect_data.py          # Records labeled hand landmark samples via webcam
├── train_model_custom.py    # Trains the Random Forest classifier on collected data
├── isl_app.py                # Real-time recognition app (prediction + text + speech)
├── isl_landmarks.csv         # Collected training data (landmark coordinates + labels)
├── isl_model_custom.pkl      # Trained model (generated after training)
└── README.md
```

## How to run

1. Install dependencies:
   ```
   pip install mediapipe opencv-python scikit-learn numpy pandas pyttsx3
   ```
2. (Optional) Collect your own training data:
   ```
   python collect_data.py
   ```
   Press a letter key (A–Z) while showing the corresponding sign to save a labeled sample. Aim for 100+ samples per letter with varied hand position for best results.
3. Train the model:
   ```
   python train_model_custom.py
   ```
4. Run the real-time app:
   ```
   python isl_app.py
   ```
   - Hold a sign steady for ~1.5 seconds to add it to the text
   - Press **S** to speak the current text aloud
   - Press **C** to clear the text
   - Press **Q** to quit

## Limitations

- **Single hand only:** the current model only tracks one hand. Indian Sign Language includes many two-handed signs, which are not yet supported.
- **Limited vocabulary:** only 10 alphabet letters are currently supported, not the full ISL alphabet or common words/phrases.
- **Signer-dependent training:** the model was trained on data from a single signer under specific lighting and camera conditions, so accuracy may vary for different users, lighting, or camera setups. In real deployment, this would need a larger, more diverse training dataset.
- **No sentence grammar or facial expressions:** real ISL grammar uses facial expressions and head movement to convey meaning (e.g. distinguishing statements from questions), which this project does not attempt to capture.
- **Regional sign variation:** Indian Sign Language has regional and personal variation in how some signs are made; this model recognizes only the specific sign style it was trained on.

## Future improvements

- Expand to the full ISL alphabet and common everyday words/phrases
- Add two-hand sign support
- Train on a larger, multi-signer dataset for better generalization
- Explore personalized/adaptive learning, where the system learns a specific user's signing style over time
- Incorporate facial expression and head-movement recognition for grammatical accuracy
- Deploy as a web app (e.g. via Streamlit) for easier access without local setup

## Disclaimer

This is a prototype built for educational and demonstration purposes. It is not a substitute for a certified human interpreter and should not be relied upon for critical or formal communication.
