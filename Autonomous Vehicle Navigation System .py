import cv2
import numpy as np
from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('lane_navigation_model.h5')

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(image, 50, 150)
    return edges

def detect_lane(image):
    edges = preprocess_image(image)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    return lines

def predict_steering_angle(image):
    processed_image = preprocess_image(image)
    processed_image = cv2.resize(processed_image, (200, 66))
    processed_image = processed_image / 255.0
    processed_image = processed_image.reshape(1, 66, 200, 1)
    steering_angle = model.predict(processed_image)[0]
    return steering_angle

@app.route('/navigate', methods=['POST'])
def navigate():
    file = request.files['image']
    image = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    lines = detect_lane(image)
    steering_angle = predict_steering_angle(image)
    
    response = {
        'steering_angle': steering_angle.tolist(),
        'lane_lines': lines.tolist() if lines is not None else []
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
