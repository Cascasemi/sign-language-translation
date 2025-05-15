from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64
import warnings
import os
import time

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Sign language labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


@app.route('/')
def index():
    return jsonify({"status": "API is running"})


@app.route('/test')
def test():
    return send_from_directory('static', 'test_local.html')


@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)
    emit('connection_status', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)


@socketio.on('frame')
def handle_frame(data):
    if 'image' not in data:
        emit('error', {'message': 'No image data received'})
        return

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Could not decode image'})
            return

        # Process frame and get annotated image
        annotated_frame, prediction_data = process_frame(frame)

        # Encode annotated frame to send back
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send both prediction and annotated frame
        emit('prediction', {
            'text': prediction_data['text'],
            'confidence': prediction_data['confidence'],
            'annotated_frame': frame_base64
        }, room=request.sid)

    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': f'Error processing image: {str(e)}'}, room=request.sid)


def process_frame(frame):
    data_aux = []
    x_ = []
    y_ = []
    prediction_data = {'text': '', 'confidence': 0}

    # Convert to RGB and process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Create a copy for annotation
    annotated_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark data for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Make prediction if model is loaded
            if model:
                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = max(prediction_proba[0])
                    predicted_character = labels_dict[int(prediction[0])]

                    prediction_data = {
                        'text': predicted_character,
                        'confidence': float(confidence)
                    }
                except Exception as e:
                    print(f"Prediction error: {e}")

    return annotated_frame, prediction_data


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)