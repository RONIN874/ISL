import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ════════════════════════════════════════════════
# 1. Load Assets & Setup
# ════════════════════════════════════════════════
MODEL_PATH = "isl_landmark_model_fixed.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODER_PATH]):
    raise FileNotFoundError("Missing model, scaler, or encoder file. Check your paths!")

print("[INFO] Loading model and preprocessing tools...")
# compile=False is used because we only need to predict, not train
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

EXPECTED_FEATURES = model.input_shape[1]
print(f"[✓] Model expects {EXPECTED_FEATURES} features per frame.")

# Map the numbers 0-25 back to the English alphabet
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# ════════════════════════════════════════════════
# 2. MediaPipe Setup
# ════════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Using stable configuration (model_complexity=1) to prevent crashing
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  
    model_complexity=1,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)

# ════════════════════════════════════════════════
# 3. Real-Time Prediction Loop
# ════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        row = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
                
        # Fix for single-hand occlusion or missing expected features
        if len(row) < EXPECTED_FEATURES:
            row.extend([0.0] * (EXPECTED_FEATURES - len(row)))
        elif len(row) > EXPECTED_FEATURES:
            row = row[:EXPECTED_FEATURES]
            
        feature_array = np.array([row])
        feature_array_scaled = scaler.transform(feature_array)
        
        predictions = model.predict(feature_array_scaled, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # ════════════════════════════════════════════════
        # 4. Map to Alphabet & Display Results
        # ════════════════════════════════════════════════
        # Get the numeric label from the encoder
        numeric_label = label_encoder.inverse_transform([predicted_idx])[0]
        
        # Convert it to an integer, then look up the letter
        letter_prediction = ALPHABET[int(numeric_label)]

        if confidence > 0.6: # 60% confidence threshold
            display_text = f"{letter_prediction} ({confidence * 100:.1f}%)"
            
            cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, display_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        
    cv2.imshow("Two-Handed ISL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()