import cv2
import os
import pandas as pd
from deepface import DeepFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense

def build_emotion_model(input_shape=(64, 64, 1), num_classes=7):
    # This architecture matches the saved weights: 2 Conv2D blocks, global pooling, and a Dense output.
    input_img = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(input_img, output)
    return model

class EmotionExtractor:
    def __init__(self, faces_dir="faces", csv_output=r"storage\results\input_video.csv"):
        self.faces_dir = faces_dir
        self.csv_output = csv_output
        self.data = []
        # Build the minimal emotion model matching the saved weights file (8 layers)
        self.emotion_model = build_emotion_model()
        self.emotion_model.load_weights("./facial_expression_model_weights.h5")

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
                analysis = DeepFace.analyze(
                    face_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    models={'emotion': self.emotion_model}
                )
                expression = analysis['dominant_emotion']
                confidence = analysis['emotion'][expression]
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
