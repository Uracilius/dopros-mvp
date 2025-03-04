import cv2
import os
import pandas as pd
from deepface import DeepFace

class EmotionExtractor:
    def __init__(self, faces_dir="faces", csv_output="storage/results/input_video.csv"):
        """
        If no csv_output is passed, it defaults to 'input_video.csv' but
        we usually override it by passing in a custom path.
        """
        self.faces_dir = faces_dir
        self.csv_output = csv_output
        self.data = []

    def extract_emotions(self):
        if not os.path.exists(self.faces_dir):
            print("Faces folder not found!")
            return

        for face_file in sorted(os.listdir(self.faces_dir)):
            if not face_file.endswith(".jpg"):
                continue

            face_path = os.path.join(self.faces_dir, face_file)
            
            # Attempt to parse the timestamp if your naming scheme includes it
            # e.g. "frame_5.jpg"
            try:
                # If your face file name is like "frame_5.jpg", parse '5' as a float
                # Adjust this parse logic if your naming differs
                timestamp_str = face_file.split("_")[1].replace(".jpg", "")
                timestamp = float(timestamp_str)
            except:
                timestamp = 0.0

            try:
                analysis = DeepFace.analyze(face_path, actions=['emotion'], enforce_detection=False)
                expression = analysis[0]['dominant_emotion']
                confidence = analysis[0]['emotion'][expression]
            except Exception as e:
                print(f"Error analyzing {face_file}: {e}")
                expression = "unknown"
                confidence = 0.0

            self.data.append([face_file, timestamp, expression, confidence])

    def save_to_csv(self):
        columns = ["Filename", "Timestamp (s)", "Expression", "Confidence"]
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv(self.csv_output, index=False)
        print(f"CSV saved: {self.csv_output}")
