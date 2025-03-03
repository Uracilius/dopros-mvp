import cv2
import os
import pandas as pd
from deepface import DeepFace
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense

def build_emotion_model(input_shape=(64, 64, 1), num_classes=7):
    model = Sequential()
    # Block 1
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(8, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 2
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Block 3
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Classification block
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

class EmotionExtractor:
    def __init__(self, faces_dir="faces", csv_output=r"storage\results\input_video.csv"):
        self.faces_dir = faces_dir
        self.csv_output = csv_output
        self.data = []
        # Build the model architecture and load the local weights
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
