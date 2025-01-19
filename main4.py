import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

# STEP ONE: DOWNLOAD DATA SET
file_path = "A_Z Handwritten Data.csv"
data = pd.read_csv(file_path)

# Use only 10,000 samples for faster training
data = data.sample(n=5000, random_state=42)

# STEP THREE: printed out the info
print("First few rows")
print(data.head())
print("Dimensions:", data.shape)
print("Column names:", data.columns)

# STEP FOUR: SPLIT AND LINEARIZE THE DATA
sample = data.iloc[0, 1:].values.reshape(28, 28)
plt.imshow(sample, cmap='gray')
plt.title(f"Label: {data.iloc[0, 0]}")
plt.show()

X = data.iloc[:, 1:].values / 255.0  # Normalize pixel values
X = X.reshape(-1, 28, 28, 1)
Y = to_categorical(data.iloc[:, 0].values, num_classes=26)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# STEP FIVE: TRAIN THE MODEL
try:
    # Try to load a saved model
    model = load_model("handwritten_character_recognition_model.h5")
    print("Model loaded successfully!")
except:
    # Create and train a new model if no saved model exists
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(26, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Save the trained model
    model.save("handwritten_character_recognition_model.h5")
    print("Model saved successfully!")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")



# END OF THE MODEL
