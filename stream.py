import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict.get('model', None)  # Ensure we get the model or handle it if missing
    if model is None:
        raise ValueError("The model is missing in the pickle file.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Label dictionary (A-Z)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'delete'}


# Streamlit UI - Sidebar Menu
st.sidebar.title("Menu")
st.sidebar.markdown(
    """
    <style>
        .sidebar-title {
            font-size: 30px;
            font-weight: bold;
            color: #074E8C;
            
        }
        .menu-item {
            font-size: 20px;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            margin: 5px 0;
            background-color: #E3F2FD;
            transition: all 0.3s ease;
        }
        .menu-item:hover {
            background-color: #BBDEFB;
            cursor: pointer;
        }
    </style>
    <div class="sidebar-title">📌 Navigation</div>
    """,
    unsafe_allow_html=True,
)

menu = st.sidebar.radio("Go to:", ["Home", "Detection Model", "Resources"])

# 🏠 Home Page
if menu == "Home":
    st.markdown(
        "<h1 style='font-size: 40px; font-family: monospace; color: #074E8C;'>Signify: Real-Time Sign Language Interpretation</h1>",
        unsafe_allow_html=True)

    # Abstract
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>📌 Abstract</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px;'>Signify is an innovative solution that uses AI and computer vision to interpret sign language into spoken or written text and vice versa in real time. It captures hand gestures, movements, and facial expressions via a camera, translating them into meaningful communication. The system supports multiple sign languages, enabling seamless interaction between hearing and non-hearing individuals, fostering inclusivity in various social and professional settings.</p>",
        unsafe_allow_html=True)

    # Problem Statement
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>⚡ Problem Statement</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px;'>How can we bridge the communication gap between individuals who use sign language and those who do not? How can technology ensure real-time, accurate, and inclusive interaction to make society more accessible for everyone?</p>",
        unsafe_allow_html=True)

    # Requirement Analysis
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>🔍 Requirement Analysis</h2>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-size: 20px;'>
        <li>High-resolution camera for capturing hand gestures and facial expressions.</li>
        <li>Robust AI models for real-time sign language recognition and translation.</li>
        <li>A user-friendly interface for seamless interaction.</li>
        <li>Support for multiple sign languages.</li>
        <li>Accurate, real-time output in text, voice, or sign animations.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Implementation
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>🛠️ Implementation</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px;'>The project involves training deep learning models on sign language datasets for gesture recognition using computer vision techniques. A user-friendly application processes real-time video input to translate sign language into text or speech and vice versa.</p>",
        unsafe_allow_html=True)

    # Future Scope
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>🚀 Future Scope</h2>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-size: 20px;'>
        <li>Expanding support for more sign languages and regional dialects.</li>
        <li>Improving accuracy using advanced AI models.</li>
        <li>Integrating with AR/VR devices for immersive communication.</li>
        <li>Applications in education, healthcare, and workplaces to enhance accessibility.</li>
    </ul>
    """, unsafe_allow_html=True)

    # Results
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>📊 Results</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px;'>The system successfully interprets sign language gestures into text or speech in real-time with high accuracy and supports seamless two-way communication. It enhances inclusivity by bridging the communication gap between hearing and non-hearing individuals in various social and professional contexts.</p>",
        unsafe_allow_html=True)

    # Conclusion
    st.markdown("<h2 style='font-size: 30px; color: #055C9D;'>✅ Conclusion</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 20px;'>The project demonstrates the potential of AI and computer vision in addressing accessibility challenges. By enabling effective communication, it fosters inclusivity, empowers the deaf community, and paves the way for a more connected and equitable society.</p>",
        unsafe_allow_html=True)


# ✋ Detection Model Page
elif menu == "Detection Model":
    st.markdown("<h1 style=' font-family: monospace; color: #074E8C;'>Sign Language Detection</h1>", unsafe_allow_html=True)
    st.write("Keep your hand visible in front of the camera to detect letters.")

    # Detected Sign Display
    st.subheader("Detected Sign:")
    detected_character_placeholder = st.empty()

    # Continuous Translation Display
    st.subheader("Continuous Translation:")
    translated_text_placeholder = st.empty()

    # Initialize session state variables
    if "output_string" not in st.session_state:
        st.session_state["output_string"] = ""
    if "last_stable_character" not in st.session_state:
        st.session_state["last_stable_character"] = None
    if "prediction_queue" not in st.session_state:
        st.session_state["prediction_queue"] = deque(maxlen=5)

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.data_aux = []
            self.x_ = []
            self.y_ = []

        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")

            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        self.x_.append(x)
                        self.y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        self.data_aux.append(x - min(self.x_))
                        self.data_aux.append(y - min(self.y_))

                x1 = int(min(self.x_) * W) - 10
                y1 = int(min(self.y_) * H) - 10

                x2 = int(max(self.x_) * W) - 10
                y2 = int(max(self.y_) * H) - 10

                # Check if the feature count matches the model's expected input
                if hasattr(model, 'n_features_in_'):
                    if len(self.data_aux) == model.n_features_in_:
                        prediction = model.predict([np.asarray(self.data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        # Add prediction to the queue
                        st.session_state["prediction_queue"].append(predicted_character)

                        # Only update the stable character if all recent predictions are the same
                        if len(st.session_state["prediction_queue"]) == st.session_state["prediction_queue"].maxlen and all(
                                p == st.session_state["prediction_queue"][0] for p in st.session_state["prediction_queue"]):
                            stable_character = st.session_state["prediction_queue"][0]
                            if stable_character != st.session_state["last_stable_character"]:
                                st.session_state["last_stable_character"] = stable_character

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    else:
                        st.error("Model does not have the expected attribute 'n_features_in_'. Check the model type.")
                        return frame

            # Handle the delete action
            if st.session_state["last_stable_character"] == 'delete':
                st.session_state["output_string"] = st.session_state["output_string"][:-1]  # Remove the last added character
                st.session_state["last_stable_character"] = None  # Reset stable character until new one is detected
                st.write(f"Detected Signs: {st.session_state['output_string']}")

            # Print the accumulated detected signs in the same line
            st.write(f"Detected Signs: {st.session_state['output_string']}")

            return frame

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# 📚 Resources Page
elif menu == "Resources":
    st.markdown("<h1 style=' font-family: monospace; color: #074E8C;'>Resources</h1>", unsafe_allow_html=True)
    st.write("Here are some useful links and references:")
    st.markdown("- 📖 [American Sign Language (ASL) Alphabet](https://www.nidcd.nih.gov/health/american-sign-language)")
    st.markdown("- 📜 [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)")
    st.markdown("- 🏗️ [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📊 [Machine Learning Model Training](https://scikit-learn.org/stable/)")
    st.markdown("- 📊 [Object detection model](https://www.youtube.com/watch?v=MJCSjXepaAM&t=1190s)")

