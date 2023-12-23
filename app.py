from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('resnet_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = img.reshape((1, 150, 150, 3)) / 255.0
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        # Use the current working directory for saving the file
        file_path = os.path.join(os.getcwd(), 'uploaded_image.jpg')
        file.save(file_path)

        # Preprocess the image
        img = preprocess_image(file_path)

        # Make a prediction
        prediction = model.predict(img)

        # Display the result
        if prediction > 0.5:
            result = "It's a dog!"
        else:
            result = "It's a cat!"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)