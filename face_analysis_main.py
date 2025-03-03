import os
import shutil
from extract_emotion import EmotionExtractor
from extract_faces import FaceExtractor

class FaceAnalysisPipeline:
    def __init__(self, model_path, video_path, faces_folder='faces'):
        """
        Initializes the face analysis pipeline.
        
        :param model_path: Path to the YOLO face detection model.
        :param video_path: Path to the input video.
        :param faces_folder: Directory where extracted faces are stored.
        """
        self.model_path = model_path
        self.video_path = video_path
        self.faces_folder = faces_folder

        self.face_extractor = FaceExtractor(self.model_path, self.video_path)
        self.emotion_extractor = EmotionExtractor()

    def run_analysis(self):
        """
        Runs the full pipeline: face extraction, emotion detection, and cleanup.
        """
        try:
            print("🔍 Extracting faces from video...")
            self.face_extractor.extract_faces()

            print("😊 Analyzing emotions...")
            self.emotion_extractor.extract_emotions()
            self.emotion_extractor.save_to_csv()

            print("🧹 Cleaning up extracted faces...")
            self.cleanup_faces()

            print("✅ Face analysis completed successfully!")

        except Exception as e:
            print(f"❌ Error in analysis pipeline: {e}")

    def cleanup_faces(self):
        """
        Deletes all files in the extracted faces folder.
        """
        if not os.path.exists(self.faces_folder):
            print(f"⚠️ Faces folder '{self.faces_folder}' does not exist, skipping cleanup.")
            return
        
        for filename in os.listdir(self.faces_folder):
            file_path = os.path.join(self.faces_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Failed to delete {file_path}. Reason: {e}")

# Example usage
if __name__ == "__main__":
    pipeline = FaceAnalysisPipeline('./yolov8n-face.pt', 'input_video.mp4')
    pipeline.run_analysis()
