import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('handwritten_character_recognition_model.h5')

def process_image(image_path):
    """Preprocess the image and predict the handwritten letter."""
    try:
        print(f"Loading image from {image_path}")
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Invalid image file or file not found.")

        # Debug: Print original image shape
        print(f"Original image shape: {img.shape}")

        # Resize, normalize, and reshape the image
        img_resized = cv2.resize(img, (28, 28))
        print(f"Resized image shape: {img_resized.shape}")

        img_normalized = img_resized / 255.0  # Normalize pixel values
        img_reshaped = img_normalized.reshape(1, 28, 28, 1)  # Reshape for model input
        print(f"Reshaped image shape: {img_reshaped.shape}")

        # Predict the letter
        print("Making prediction...")
        predicted_class = model.predict(img_reshaped)
        print(f"Raw model prediction: {predicted_class}")

        predicted_label = np.argmax(predicted_class, axis=1)
        return chr(predicted_label[0] + 65)  # Convert label to letter
    except Exception as e:
        print(f"Error during image processing: {e}")
        raise ValueError("Error during image processing")

# Path to the test image
image_path = 'm.png'  # Ensure 'm.png' is in the same directory as this script

# Test the function
try:
    predicted_letter = process_image(image_path)
    print(f"Predicted letter: {predicted_letter}")
except ValueError as e:
    print(f"Test failed: {e}")
