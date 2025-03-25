import os
import mediapipe as mp

class EmotionDatasetProcessor:
    def __init__(self, dataset_dir, saved_np_dir, chunk_size=2000):
        self.dataset_dir = dataset_dir
        self.saved_np_dir = saved_np_dir
        self.chunk_size = chunk_size
        os.makedirs(saved_np_dir, exist_ok=True)
        self.target_labels = ['neutral','angry','sad','happy']
        self.label_map = {n:i for i,n in enumerate(self.target_labels)}
        self.num_classes = len(self.target_labels)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def extract_and_save_chunks(self):
        pass

    def merge_chunks_and_save(self):
        pass