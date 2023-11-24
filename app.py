from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
import numpy as np

facenet_model = InceptionResNetV2(weights='imagenet', include_top=True, input_shape=(299, 299, 3))

def get_face_from_image(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for i, (x, y, w, h) in enumerate(faces):
        face = img[y:y+h, x:x+w]
        break
    return face

def preprocess_image(img):
    img = cv2.resize(img, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def get_face_embeddings(img):
    preprocessed_image = preprocess_image(img)
    embeddings = facenet_model.predict(preprocessed_image)
    return embeddings.flatten()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

app = Flask(__name__)

def process_image(image):
    try:
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR) 
        return image
    except Exception as e:
        raise RuntimeError(f'Error processing image: {str(e)}')

@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'})

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return jsonify({'error': 'Both files must be selected'})

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if ('.' not in file1.filename or file1.filename.rsplit('.', 1)[1].lower() not in allowed_extensions or
        '.' not in file2.filename or file2.filename.rsplit('.', 1)[1].lower() not in allowed_extensions):
        return jsonify({'error': 'Invalid file type'})

    try:
        img_1 = process_image(file1)
        img_2 = process_image(file2)
        face_1 = get_face_from_image(img_1)
        face_2 = get_face_from_image(img_2)

        embeddings1 = get_face_embeddings(face_1)
        embeddings2 = get_face_embeddings(face_2)

        similarity = cosine_similarity(embeddings1, embeddings2)
        
        if similarity >= 0.5:
            return jsonify({'similar': True})
        else:
            return jsonify({'similar': False})
            
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
