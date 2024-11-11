from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
import requests
from tensorflow.keras.models import load_model

app = Flask(__name__)
# Load trained model
model = load_model('E:/pythonproject/pythonProject/pythonproject/Face/model/classifier_elon_zurk.h5')

# Recognize faces
def recognize_faces(image):
    # Load faces recognition model from opencv
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # make sure is not None
    if image is None:
        raise ValueError("Image could not be decoded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    return faces

# Prediction function
def predict_person(face_image):
    face_image_resized = cv2.resize(face_image, (224, 224))  # Input size
    face_image_normalized = face_image_resized / 255.0  # Normalization
    face_image_reshaped = np.reshape(face_image_normalized, (1, 224, 224, 3))

    predictions = model.predict(face_image_reshaped)
    return predictions

@app.route('/recognize_faces', methods=['POST'])
def recognize_faces_api():
    data = request.json
    image_urls = data.get('image_urls', [])
    results = []

    for url in image_urls:
        # Download picture from the image url
        response = requests.get(url)
        print(f"Downloading image from {url}, status code: {response.status_code}")  # Status code
        if response.status_code == 200:
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

            if image is not None:
                faces = recognize_faces(image)

                for (x, y, w, h) in faces:
                    # Recognize all faces from the input image
                    face_image = image[y:y + h, x:x + w]

                    # Prediction to the face
                    predictions = predict_person(face_image)

                    person_name = None
                    # Determine the result: If neither Musk nor Zuckerberg, do nothing
                    if predictions[0][0] > 0.8:  # The first one is Elon Musk
                        person_name = "Elon Musk"
                    elif predictions[0][1] > 0.8:  # The second one is Mark Zuckerberg
                        person_name = "Mark Zuckerberg"

                    # Draw rectangle around all the faces in the image, and tag Musk nor Zuckerberg
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
                    cv2.putText(image, person_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # Save and output
                _, buffer = cv2.imencode('.jpg', image)
                image_stream = io.BytesIO(buffer)
                return send_file(image_stream, mimetype='image/jpeg')
            else:
                results.append({"url": url, "error": "Image could not be decoded."})
        else:
            results.append({"url": url, "error": "Failed to download image."})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
