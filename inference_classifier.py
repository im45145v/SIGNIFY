import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Load the pre-trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict.get('model', None)  # Ensure we get the model or handle it if missing
    if model is None:
        raise ValueError("The model is missing in the pickle file.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'delete'}

# Variables for smoothing predictions
prediction_queue = deque(maxlen=5)  # Store last 5 predictions
last_stable_character = None
output_string = ""  # Final output string

last_added_character = None  # To keep track of the last added character to output_string

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

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

        for hand_landmarks in results. multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Check if the feature count matches the model's expected input
        if hasattr(model, 'n_features_in_'):
            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Add prediction to the queue
                prediction_queue.append(predicted_character)

                # Only update the stable character if all recent predictions are the same
                if len(prediction_queue) == prediction_queue.maxlen and all(
                        p == prediction_queue[0] for p in prediction_queue):
                    stable_character = prediction_queue[0]
                    if stable_character != last_stable_character:
                        last_stable_character = stable_character

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        else:
            print("Model does not have the expected attribute 'n_features_in_'. Check the model type.")
            break

    # Handle the delete action
    if last_stable_character == 'delete':
        if last_added_character is not None:
            output_string = output_string[:-1]  # Remove the last added character
            last_added_character = None  # Reset the last added character
        last_stable_character = None  # Reset stable character until new one is detected
        print("\rDetected Signs: " + output_string, end="")

    # Print the accumulated detected signs in the same line
    print("\rDetected Signs: " + output_string, end="")

    cv2.imshow('frame', frame)

    # Check for 'q' key press to append the detected sign to the main string
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and last_stable_character is not None and last_stable_character != 'delete':
        output_string += last_stable_character  # Append the last detected character to the output string
        last_added_character = last_stable_character  # Store the last added character
        print("\rDetected Signs: " + output_string, end="")  # Print updated output string

    # Break loop if 'ESC' key is pressed
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()