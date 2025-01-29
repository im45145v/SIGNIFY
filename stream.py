import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import deque
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

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

menu = st.sidebar.radio("", ["Home", "Detection Model", "Resources"])

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

    # Start Camera
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected. Please check your permissions.")
            break

        # Flip frame for mirrored effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe Hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Collect landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Initialize delete cooldown timestamp in session state
            if "last_delete_time" not in st.session_state:
                st.session_state["last_delete_time"] = 0  # Start at 0 so first delete can happen

            # Make prediction
            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Stability check
                st.session_state["prediction_queue"].append(predicted_character)
                if len(st.session_state["prediction_queue"]) == st.session_state["prediction_queue"].maxlen and all(
                        p == st.session_state["prediction_queue"][0] for p in st.session_state["prediction_queue"]
                ):
                    stable_character = st.session_state["prediction_queue"][0]

                    # Handle delete with cooldown
                    current_time = time.time()
                    if stable_character == "delete":
                        if current_time - st.session_state["last_delete_time"] >= 5:  # 5-second cooldown
                            st.session_state["output_string"] = st.session_state["output_string"][
                                                                :-1]  # Remove last character
                            st.session_state["last_delete_time"] = current_time  # Update last delete time
                    elif stable_character != st.session_state["last_stable_character"]:
                        repeat_threshold = 1
                        st.session_state["output_string"] += stable_character * repeat_threshold
                        st.session_state["last_stable_character"] = stable_character

        # Update placeholders
        detected_character_placeholder.write(f"**{st.session_state.get('last_stable_character', 'Waiting for detection...')}**")
        translated_text_placeholder.write(f"{st.session_state['output_string']}")

        # Resize frame for a larger display
        frame_resized = cv2.resize(frame, (800, 600))  # Adjust width and height as needed
        FRAME_WINDOW.image(frame_resized, channels="BGR")

        # Exit condition
        if st.session_state.get("exit_app"):
            break

    cap.release()

# 📚 Resources Page
elif menu == "Resources":
    st.markdown("<h1 style=' font-family: monospace; color: #074E8C;'>Resources</h1>", unsafe_allow_html=True)
    st.write("Here are some useful links and references:")
    st.markdown("- 📖 [American Sign Language (ASL) Alphabet](https://www.nidcd.nih.gov/health/american-sign-language)")
    st.markdown("- 📜 [MediaPipe Hands Documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)")
    st.markdown("- 🏗️ [Streamlit Documentation](https://docs.streamlit.io/)")
    st.markdown("- 📊 [Machine Learning Model Training](https://scikit-learn.org/stable/)")
    st.markdown("- 📊 [Object detection model](https://www.youtube.com/watch?v=MJCSjXepaAM&t=1190s)")

