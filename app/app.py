import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os

# Add project root and src folder to system path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

from src.emotion_detector import EmotionDetector
from src.prompt_builder import build_prompt
from src.chat_engine import ChatEngine

def main():
    st.title("Emotion-Aware Chatbot")

    # Initialize emotion detector and chat engine
    emotion_detector = EmotionDetector()
    chat_engine = ChatEngine()

    st.write("Upload an image of your face to detect your current emotion, then chat with the bot.")

    uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])

    detected_emotion = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)[:, :, ::-1]  # Convert RGB to BGR for OpenCV

        detected_emotion = emotion_detector.detect_emotion(image_np)
        if detected_emotion:
            st.success(f"Detected Emotion: {detected_emotion}")
        else:
            st.warning("No face detected or unable to detect emotion.")

        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Text input for chat
    user_input = st.text_input("Enter your message to the chatbot:")

    if user_input:
        prompt = build_prompt(user_input, detected_emotion)
        response = chat_engine.get_response(prompt)
        st.markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()
