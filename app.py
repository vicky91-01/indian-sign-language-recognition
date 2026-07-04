import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from gtts import gTTS
import io
from streamlit_autorefresh import st_autorefresh

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="ISL Sign Recognition", page_icon="🤟", layout="wide")

st.title("🤟 Real-Time Indian Sign Language Recognition")
st.markdown(
    "Show a hand sign to your webcam. Hold it steady for about 1.5 seconds to "
    "add the letter to your sentence. Currently supports: **A, B, C, D, K, L, O, V, W, Y**"
)

# ---------- LOAD MODEL (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return joblib.load("isl_model_custom.pkl")

model = load_model()

HOLD_DURATION = 1.5
CONFIDENCE_THRESHOLD = 0.6

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class SignPredictor(VideoProcessorBase):
    """
    IMPORTANT: This class runs in a separate background thread.
    It must NOT touch st.session_state directly (that causes the
    stream to silently freeze/crash). All state is kept as plain
    instance attributes instead, and read safely from the main
    thread afterwards.
    """
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.text = ""
        self.last_letter = None
        self.letter_start_time = None
        self.predicted_letter = None
        self.confidence = 0.0
        self.clear_requested = False
        self.frame_count = 0
        self.last_display_letter = None
        self.last_display_confidence = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Downscale for faster MediaPipe processing, then scale drawing back up
        small_img = cv2.resize(img, (0, 0), fx=0.6, fy=0.6)

        if self.clear_requested:
            self.text = ""
            self.clear_requested = False

        self.frame_count += 1
        # Only run the (expensive) hand detection + prediction every 2nd frame.
        # On skipped frames we just redraw the last known result, which keeps
        # the video feed smooth without doubling the CPU load.
        run_detection = (self.frame_count % 2 == 0)

        predicted_letter = self.last_display_letter
        confidence = self.last_display_confidence

        try:
            if run_detection:
                rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)

                predicted_letter = None
                confidence = 0.0

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        landmark_list = []
                        for lm in hand_landmarks.landmark:
                            landmark_list.extend([lm.x, lm.y, lm.z])

                        X_input = np.array(landmark_list).reshape(1, -1)
                        probabilities = model.predict_proba(X_input)[0]
                        best_idx = np.argmax(probabilities)
                        confidence = float(probabilities[best_idx])
                        predicted_letter = model.classes_[best_idx]

                self.last_display_letter = predicted_letter
                self.last_display_confidence = confidence

            # ---------- HOLD-TO-CONFIRM LOGIC (instance state only) ----------
            if predicted_letter is not None and confidence >= CONFIDENCE_THRESHOLD:
                if predicted_letter == self.last_letter:
                    if self.letter_start_time is not None:
                        elapsed = time.time() - self.letter_start_time
                        if elapsed >= HOLD_DURATION:
                            self.text += predicted_letter
                            self.last_letter = None
                            self.letter_start_time = None
                else:
                    self.last_letter = predicted_letter
                    self.letter_start_time = time.time()
            else:
                self.last_letter = None
                self.letter_start_time = None

            self.predicted_letter = predicted_letter
            self.confidence = confidence

            # ---------- DRAW INFO ON FRAME ----------
            if predicted_letter is not None:
                color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
                cv2.putText(img, f"{predicted_letter} ({confidence*100:.0f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                if self.last_letter == predicted_letter and self.letter_start_time:
                    elapsed = time.time() - self.letter_start_time
                    progress = min(elapsed / HOLD_DURATION, 1.0)
                    bar_width = int(250 * progress)
                    cv2.rectangle(img, (10, 55), (10 + 250, 70), (100, 100, 100), 2)
                    cv2.rectangle(img, (10, 55), (10 + bar_width, 70), (0, 255, 0), -1)
            else:
                cv2.putText(img, "No hand detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        except Exception:
            # Never let an error inside recv() kill the stream —
            # just return the raw frame if something goes wrong.
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:openrelay.metered.ca:80"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
                {
                    "urls": ["turn:openrelay.metered.ca:443"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
                {
                    "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                    "username": "openrelayproject",
                    "credential": "openrelayproject",
                },
            ]
        }
    )
    ctx = webrtc_streamer(
        key="isl-recognition",
        video_processor_factory=SignPredictor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}, "frameRate": {"ideal": 15}},
            "audio": False,
        },
    )

with col2:
    st.subheader("Your text")
    text_display = st.empty()
    speak_area = st.empty()

    col_a, col_b = st.columns(2)
    with col_a:
        speak_clicked = st.button("🔊 Speak", use_container_width=True)
    with col_b:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    current_text = ""
    if ctx.video_processor:
        current_text = ctx.video_processor.text

        if clear_clicked:
            ctx.video_processor.clear_requested = True
            current_text = ""
            st.session_state.pop("audio_bytes", None)

        if speak_clicked:
            if current_text:
                try:
                    with st.spinner("Generating speech..."):
                        tts = gTTS(text=current_text, lang="en")
                        audio_buffer = io.BytesIO()
                        tts.write_to_fp(audio_buffer)
                        audio_buffer.seek(0)
                        st.session_state.audio_bytes = audio_buffer.read()
                except Exception as e:
                    st.session_state.pop("audio_bytes", None)
                    speak_area.error(f"Couldn't generate speech: {e}")
            else:
                st.session_state.pop("audio_bytes", None)
                speak_area.warning("No text yet — sign a few letters first.")

    # Re-render the audio player on every rerun (including auto-refresh
    # ticks) as long as we have generated audio, so it doesn't vanish
    # before it gets a chance to play.
    if st.session_state.get("audio_bytes"):
        speak_area.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)

    text_display.markdown(f"### {current_text or '_(nothing yet)_'}")

    st.divider()
    st.caption(
        "⚠️ This is a prototype for educational/demo purposes, trained on a small "
        "self-collected dataset (10 letters). Accuracy may vary with lighting, "
        "camera angle, and signing style. Not a substitute for a certified interpreter."
    )

    # Auto-refresh the page every 500ms so the text box stays in sync
    # with the background video thread, without needing manual clicks.
    if ctx.state.playing:
        st_autorefresh(interval=800, key="text_refresh")

st.divider()
st.markdown(
    "Built with MediaPipe hand-landmark detection + Random Forest classification. "
    "[View source on GitHub](https://github.com/vicky91-01/indian-sign-language-recognition)"
)
