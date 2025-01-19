import time
from tensorflow.keras.models import load_model

start_time = time.time()
model = load_model('handwritten_character_recognition_model.h5')
print(f"Model loaded in {time.time() - start_time:.2f} seconds")
