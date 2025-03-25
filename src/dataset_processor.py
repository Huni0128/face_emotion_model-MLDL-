import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import gc

from tqdm import tqdm
from sklearn.model_selection import train_test_split


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
        if any(f.startswith('img_chunk_') for f in os.listdir(self.saved_np_dir)):
            print("이미 chunk파일이 존재하여 추출을 건너뜁니다.")
            return

        chunk_id = 0
        for label in sorted(os.listdir(self.dataset_dir)):
            if label not in self.target_labels:
                continue

            files = [f for f in os.listdir(os.path.join(self.dataset_dir, label))
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

            for start in range(0, len(files), self.chunk_size):
                images, landmarks, labels = [], [], []
                for fname in tqdm(files[start:start+self.chunk_size], desc=f"Processing {label}"):
                    img_path = os.path.join(self.dataset_dir, label, fname)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)
                    resized = cv2.resize(gray, (64,64)) / 255.0

                    results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if results.multi_face_landmarks:
                        lm = [coord for lmks in results.multi_face_landmarks
                              for pt in lmks.landmark for coord in (pt.x, pt.y, pt.z)]
                        if len(lm) == 1404:
                            images.append(resized[..., None])
                            landmarks.append(lm)
                            labels.append(self.label_map[label])

                if images:
                    np.save(os.path.join(self.saved_np_dir, f"img_chunk_{chunk_id}.npy"),
                            np.array(images, dtype=np.float32))
                    np.save(os.path.join(self.saved_np_dir, f"lm_chunk_{chunk_id}.npy"),
                            np.array(landmarks, dtype=np.float32))
                    np.save(os.path.join(self.saved_np_dir, f"label_chunk_{chunk_id}.npy"),
                            tf.keras.utils.to_categorical(labels, self.num_classes))
                    print(f"Saved chunk {chunk_id} ({len(images)} samples)")
                    chunk_id += 1
                gc.collect()

    def merge_chunks_and_save(self):
        pass

if __name__ == "__main__":
    processor = EmotionDatasetProcessor(dataset_dir="data", saved_np_dir="saved_np")
    processor.extract_and_save_chunks()
    processor.merge_chunks_and_save()
