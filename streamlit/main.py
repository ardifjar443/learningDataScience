import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

# Load the saved model
model = load_model('model/best_model.h5')

st.title('Emotion Detection App')

# Helper function to preprocess image
def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to predict emotion from image
def predict_emotion(img_array):
    predictions = model.predict(img_array)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_emotion = emotions[np.argmax(predictions)]
    return predicted_emotion

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)

        # Resize image to match the model input size
        img_resized = img_pil.resize((48, 48))

        # Preprocess the image
        img_array = preprocess_image(img_resized)

        # Make prediction
        predicted_emotion = predict_emotion(img_array)

        # Display the image and predict
