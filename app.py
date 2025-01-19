from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('handwritten_character_recognition_model.h5')

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save and process the image
    image_path = 'uploaded_image.png'
    image_file.save(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    # Predict the letter
    predicted_class = model.predict(img_reshaped)
    predicted_label = np.argmax(predicted_class, axis=1)
    letter = chr(predicted_label[0] + 65)

    return jsonify({"predicted_letter": letter})

if __name__ == '__main__':
    app.run(debug=True, port=8080)



def check(uploaded_image_path):
    """Process the image and predict the letter."""
    try:
        # Load the image
        img = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        if img is None:
            raise ValueError("Invalid image file or file not found.")

        # Preprocess the image: resize to 28x28 pixels and normalize
        img_resized = cv2.resize(img, (28, 28))  # Resize the image to 28x28 pixels
        img_normalized = img_resized / 255.0  # Normalize the pixel values (scale between 0 and 1)

        # Reshape the image to (1, 28, 28, 1) to match the model's input shape
        img_reshaped = img_normalized.reshape(1, 28, 28, 1)

        # Make the prediction
        predicted_class = model.predict(img_reshaped)
        predicted_label = np.argmax(predicted_class, axis=1)  # Get the class index (0-25)

        # Map the predicted class index to a letter (0 -> 'A', 1 -> 'B', ..., 25 -> 'Z')
        letter = chr(predicted_label[0] + 65)  # Convert the index to the corresponding letter

        # Print the predicted letter
        print(f"Predicted letter: {letter}")
        return letter

    except Exception as e:
        print(f"Error during image processing: {e}")
        return None


# Example usage
uploaded_image_path = 'uploaded_image.png'  # Replace with your image file path

if uploaded_image_path is not None:
    predicted_letter = check(uploaded_image_path)
    if predicted_letter:
        print(f"The identified letter is: {predicted_letter}")
    else:
        print("Failed to identify the letter.")
else:
    print("No image file provided.")
