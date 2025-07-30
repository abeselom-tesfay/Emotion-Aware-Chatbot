import streamlit as st
from PIL import Image
import numpy as np
import cv2
import sys
import os

# Path setup
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
sys.path.append(ROOT_DIR)
sys.path.append(SRC_DIR)

from src.emotion_detector import EmotionDetector
from src.prompt_builder import build_prompt
from src.chat_engine import ChatEngine

def main():
    st.set_page_config(page_title="Emotion-Aware Chatbot", layout="centered")
    st.title("Emotion-Aware Chatbot")

    emotion_detector = EmotionDetector()
    chat_engine = ChatEngine()

    st.write("Upload your face image to detect your emotion. Then chat and receive tailored emotional responses.")

    uploaded_file = st.file_uploader("Upload your face image", type=["jpg", "jpeg", "png"])

    detected_emotion = None
    image = None

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)[:, :, ::-1] 
        detected_emotion = emotion_detector.detect_emotion(image_np)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if detected_emotion:
            st.success(f"Detected Emotion: {detected_emotion}")
        else:
            st.warning("No face detected or unable to detect emotion.")

    user_input = st.text_input("Enter your message:")

    if user_input:
        prompt = build_prompt(user_input, detected_emotion)
        response = chat_engine.get_response(prompt)
        st.markdown(f"**Chatbot Response:** {response}")

if __name__ == "__main__":
    main()
