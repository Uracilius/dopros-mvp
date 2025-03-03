import cv2
import os
import pandas as pd
from deepface import DeepFace

class EmotionExtractor:
    def __init__(self, faces_dir="faces", csv_output="faces_data.csv"):
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
            timestamp = float(face_file.split("_")[1].replace(".jpg", ""))

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

if __name__ == "__main__":
    extractor = EmotionExtractor()
    extractor.extract_emotions()
    extractor.save_to_csv()
