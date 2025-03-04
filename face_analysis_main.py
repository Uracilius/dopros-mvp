import os
import shutil
from extract_emotion import EmotionExtractor
from extract_faces import FaceExtractor

import hashlib
import os

def hashed_filename(video_path: str, prefix: str = "video_", suffix: str = ".csv") -> str:
    """
    Returns a hashed filename (SHA-256) based on the original video filename.
    For example, 'sample_video.mp4' -> 'video_516f2eb...something...d52.csv'
    """
    # Extract just the basename, e.g. 'sample_video.mp4'
    base_name = os.path.basename(video_path)
    
    # Generate the SHA-256 hash of the base_name
    hash_object = hashlib.sha256(base_name.encode('utf-8'))
    hex_digest = hash_object.hexdigest()
    
    # Form a final string, e.g. 'video_[hash].csv'
    hashed_name = f"{prefix}{hex_digest}{suffix}"
    
    return hashed_name

class FaceAnalysisPipeline:
    def __init__(self, model_path, video_path, faces_folder='faces'):
        """
        Initializes the face analysis pipeline with a dynamic CSV output name.
        """
        self.model_path = model_path
        self.video_path = video_path
        self.faces_folder = faces_folder

        # Dynamically compute CSV name based on the video filename
        self.csv_output_path = os.path.join(
            "storage/results",
            hashed_filename(video_path, prefix="video_", suffix=".csv")
        )

        self.face_extractor = FaceExtractor(self.model_path, self.video_path, output_dir=self.faces_folder)
        self.emotion_extractor = EmotionExtractor(
            faces_dir=self.faces_folder,
            csv_output=self.csv_output_path
        )

    def run_analysis(self):
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

    def get_csv_output_path(self) -> str:
        """
        Returns the absolute path to the final hashed CSV file
        so that other parts of the code can reference it if needed.
        """
        return self.csv_output_path

# Example usage
if __name__ == "__main__":
    pipeline = FaceAnalysisPipeline('./yolov8n-face.pt', 'input_video.mp4')
    pipeline.run_analysis()
