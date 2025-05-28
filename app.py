from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
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

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'

# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://fit-firstly-tuna.ngrok-free.app",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure Socket.IO
socketio = SocketIO(app,
                    cors_allowed_origins=[
                        "https://fit-firstly-tuna.ngrok-free.app",
                        "http://localhost:5000",
                        "http://127.0.0.1:*"
                    ],
                    logger=True,
                    engineio_logger=True
                    )

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
mp_selfie_segmentation = mp.solutions.selfie_segmentation

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# Initialize selfie segmentation for background removal
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Global background image
background_image = None


def load_background_image(image_path, width=640, height=480):
    """Load and resize background image"""
    global background_image
    try:
        bg = cv2.imread(image_path)
        if bg is not None:
            background_image = cv2.resize(bg, (width, height))
            print(f"Background image loaded: {image_path}")
            return True
        else:
            print(f"Could not load background image: {image_path}")
            return False
    except Exception as e:
        print(f"Error loading background image: {e}")
        return False


# Load default background (you can change this path)
load_background_image('./background.jpg')  # Place your background image here


@app.route('/')
def index():
    return jsonify({"status": "API is running"})


@app.route('/test')
def test():
    return send_from_directory('static', 'test_local.html')


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "background_loaded": background_image is not None,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })


@app.route('/upload_background', methods=['POST'])
def upload_background():
    """Upload a new background image"""
    try:
        if 'background' not in request.files:
            return jsonify({'error': 'No background file provided'}), 400

        file = request.files['background']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save uploaded file
        filename = 'uploaded_background.jpg'
        filepath = os.path.join('.', filename)
        file.save(filepath)

        # Load the new background
        if load_background_image(filepath):
            return jsonify({'message': 'Background updated successfully'})
        else:
            return jsonify({'error': 'Failed to load background image'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.after_request
def after_request(response):
    """Add additional CORS headers"""
    response.headers.add('Access-Control-Allow-Origin', 'https://fit-firstly-tuna.ngrok-free.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')
    emit('connection_status', {
        'status': 'connected',
        'sid': request.sid,
        'server_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'background_available': background_image is not None
    })


@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')


@socketio.on('frame')
def handle_frame(data):
    if 'image' not in data:
        emit('error', {'message': 'No image data received'}, room=request.sid)
        return

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': 'Could not decode image'}, room=request.sid)
            return

        # Get background replacement preference
        use_background_replacement = data.get('background_replacement', True)

        # Process frame
        annotated_frame, prediction_data = process_frame(frame, use_background_replacement)

        # Encode annotated frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send response
        emit('prediction', {
            'text': prediction_data['text'],
            'confidence': prediction_data['confidence'],
            'annotated_frame': frame_base64,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, room=request.sid)

    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {
            'message': f'Error processing image: {str(e)}',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }, room=request.sid)


def apply_background_replacement(frame):
    """Apply background replacement using MediaPipe selfie segmentation"""
    global background_image

    if background_image is None:
        return frame

    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to get segmentation mask
        results = selfie_segmentation.process(rgb_frame)

        # Create mask
        mask = results.segmentation_mask

        # Resize background to match frame size
        h, w = frame.shape[:2]
        bg_resized = cv2.resize(background_image, (w, h))

        # Create 3-channel mask
        mask_3channel = np.stack((mask,) * 3, axis=-1)

        # Apply threshold to create binary mask
        mask_3channel = (mask_3channel > 0.5).astype(np.float32)

        # Blend foreground and background
        result = frame * mask_3channel + bg_resized * (1 - mask_3channel)

        return result.astype(np.uint8)

    except Exception as e:
        print(f"Error in background replacement: {e}")
        return frame


def process_frame(frame, use_background_replacement=True):
    data_aux = []
    x_ = []
    y_ = []
    prediction_data = {'text': '', 'confidence': 0}

    # Apply background replacement first if enabled
    if use_background_replacement:
        frame = apply_background_replacement(frame)

    # Convert to RGB and process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Create a copy for annotation
    annotated_frame = frame.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Only add to data_aux if we have landmarks
            if x_ and y_:
                min_x, min_y = min(x_), min(y_)
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

                # Make prediction if model is loaded and we have enough data
                if model and len(data_aux) >= 42:  # 21 landmarks * 2 coordinates
                    try:
                        prediction = model.predict([np.asarray(data_aux[:42])])
                        prediction_proba = model.predict_proba([np.asarray(data_aux[:42])])
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
    print(f"Starting server on port {port}")
    print(f"Ngrok URL: https://fit-firstly-tuna.ngrok-free.app")
    print("Place your background image as 'background.jpg' in the same directory")
    socketio.run(app,
                 host='0.0.0.0',
                 port=port,
                 debug=True
                 )