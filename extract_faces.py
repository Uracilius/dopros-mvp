import cv2
import os

class FaceExtractor:
    def __init__(self, model_path, video_path, output_dir="faces"):
        self.model_path = model_path
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_skip = max(1, self.fps // 2)  # every ~0.5 seconds
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_faces(self):
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # Save a frame every 'frame_skip' frames
            if frame_count % self.frame_skip == 0:
                filename = os.path.join(self.output_dir, f"frame_{saved_count}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

        self.cap.release()
        print(f"Saved {saved_count} frames to '{self.output_dir}'.")
