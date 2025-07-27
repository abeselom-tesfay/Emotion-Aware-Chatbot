import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path='models/emotion_model.h5'):
        self.model = load_model(model_path)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        # Load OpenCV's pre-trained face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, image):
        """
        Detect face and predict emotion in the given image.
        Args:
            image (np.array): BGR image from OpenCV or loaded from PIL
        Returns:
            (str) predicted emotion label or None if no face found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            return None  # No face detected

        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)  

        preds = self.model.predict(roi_gray)[0]
        emotion_idx = np.argmax(preds)
        return self.emotion_labels[emotion_idx]
