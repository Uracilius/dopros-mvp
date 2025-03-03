import cv2
from ultralytics import YOLO
import os

class FaceExtractor:
    def __init__(self, model_path, video_path, output_dir="faces"):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_skip = max(1, self.fps // 2)
        os.makedirs(output_dir, exist_ok=True)
        self.frame_count = 0
        self.face_count = 0

    def extract_faces(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            if self.frame_count % self.frame_skip != 0:
                continue

            timestamp = self.frame_count / self.fps
            timestamp_str = f"{timestamp:.2f}"

            results = self.model(frame)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face_filename = f"{self.output_dir}/face_{timestamp_str}.jpg"
                        cv2.imwrite(face_filename, face)
                        self.face_count += 1

        self.cap.release()
        print(f"Extracted {self.face_count} faces from video, processing every {self.frame_skip} frames.")

# Usage
if __name__ == "__main__":
    face_extractor = FaceExtractor("yolov8n-face.pt", "input_video.mp4")
    face_extractor.extract_faces()
