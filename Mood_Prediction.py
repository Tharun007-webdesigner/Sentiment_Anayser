from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model(r"C:\tharun\Datasets\face recognition\keras_model.h5", compile=False)

# Load the labels
class_names = open(r"C:\tharun\Datasets\face recognition\labels.txt", "r").readlines()

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 corresponds to the default webcam

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Convert the frame into a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize and crop the image
    size = (224, 224)
    image = ImageOps.fit(pil_image, size, Image.Resampling.LANCZOS)

    # Convert the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the frame along with the predictions
    cv2.putText(frame, f"Class: {class_name[2:]}, Confidence Score: {confidence_score:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Webcam Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
