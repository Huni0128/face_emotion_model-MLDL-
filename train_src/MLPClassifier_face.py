import pandas as pd
import os
from deepface import DeepFace  # 얼굴 임베딩 추출을 위한 DeepFace 라이브러리
from sklearn.model_selection import train_test_split  # 학습/테스트 데이터 분할 도구
from sklearn.neural_network import MLPClassifier  # 다층 퍼셉트론 분류 모델
from sklearn.metrics import classification_report  # 분류 결과 평가 리포트 생성
import numpy as np
from tqdm import tqdm
import joblib  # 학습된 모델 저장/로드를 위한 라이브러리

# 숫자 라벨을 텍스트 감정 라벨로 매핑하는 딕셔너리 정의
label_map = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

# 임베딩과 라벨 파일을 저장할 폴더 경로 설정
BASE_DIR = "./MLPClassifier"
os.makedirs(BASE_DIR, exist_ok=True)

# 최종 저장 파일 경로와 중간 저장 파일 경로 설정
embeddings_file = os.path.join(BASE_DIR, "embeddings.npy")
labels_file = os.path.join(BASE_DIR, "labels.npy")
intermediate_embeddings = os.path.join(BASE_DIR, "embeddings_intermediate.npy")
intermediate_labels = os.path.join(BASE_DIR, "labels_intermediate.npy")

def load_dataset_from_csv(csv_path, img_root, model_name="ArcFace", # model_name: 사용할 얼굴 임베딩 모델 이름
                          batch_size=5000, 
                          embeddings_file=embeddings_file, 
                          labels_file=labels_file,
                          intermediate_embeddings=intermediate_embeddings,
                          intermediate_labels=intermediate_labels):
    df = pd.read_csv(csv_path)
    # "path" 컬럼이 있으면 "filename"으로 이름 변경
    if "path" in df.columns:
        df.rename(columns={"path": "filename"}, inplace=True)
    # "filename" 컬럼이 없으면 첫번째 컬럼은 파일명, 두번째 컬럼은 라벨로 간주
    elif "filename" not in df.columns:
        df.columns = ["filename", "label"]
    
    X, y = [], []  # 임베딩과 라벨을 저장할 리스트
    total = len(df)
    
    # 중간 저장 파일이 존재하면 이어서 진행
    if os.path.exists(intermediate_embeddings) and os.path.exists(intermediate_labels):
        print("중간 임베딩 파일을 로드합니다...")
        X = list(np.load(intermediate_embeddings, allow_pickle=True))
        y = list(np.load(intermediate_labels, allow_pickle=True))
        start_index = len(X)  # 이미 처리된 데이터 수 만큼 건너뜀
    else:
        start_index = 0

    # CSV 데이터셋을 배치 단위로 처리
    for start in tqdm(range(start_index, total, batch_size), desc="배치 처리"):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]  # 배치에 해당하는 부분 데이터프레임 추출
        for _, row in batch_df.iterrows():
            # CSV에 저장된 파일 경로가 전체 경로라면 basename만 사용
            filename = os.path.basename(row["filename"])
            # 이미지 파일 경로 생성
            img_path = os.path.join(img_root, filename)
            if not os.path.exists(img_path):
                # 이미지 파일이 존재하지 않으면 오류 메시지 출력
                print(f"오류: {img_path} → 파일이 존재하지 않습니다.")
                continue
            try:
                # DeepFace를 사용하여 이미지에서 임베딩을 추출
                embedding = DeepFace.represent(img_path=img_path,
                                               model_name=model_name,
                                               enforce_detection=False)[0]['embedding']
                X.append(embedding)  # 임베딩 추가
                y.append(label_map[int(row["label"])])  # 라벨 매핑 후 추가
            except Exception as e:
                # 임베딩 추출 과정에서 오류 발생 시 오류 메시지 출력
                print(f"오류: {img_path} → {e}")
                continue
        # 각 배치가 끝난 후 중간 결과를 저장하여 작업 중간에 중단되어도 이어서 작업 가능하게 함
        np.save(intermediate_embeddings, np.array(X, dtype=object))
        np.save(intermediate_labels, np.array(y, dtype=object))
        print(f"배치 [{start}:{end}] 완료, 중간 저장됨.")

    # 최종적으로 리스트를 배열로 변환하여 저장 
    X = np.array(X, dtype=object)
    y = np.array(y, dtype=object)
    np.save(embeddings_file, X)
    np.save(labels_file, y)
    print("임베딩 추출 및 최종 저장 완료!")
    return X, y

# CSV 파일 경로와 이미지 폴더 경로를 설정
csv_path = "./src/crops_dataset.csv"
img_root = "./crops_dataset"

# 위에서 정의한 load_dataset_from_csv 함수를 호출하여 임베딩과 라벨 데이터를 로드 및 임베딩 추출
X, y = load_dataset_from_csv(csv_path, img_root, batch_size=5000)

# 학습 데이터와 테스트 데이터를 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# MLPClassifier를 사용하여 분류 모델을 학습
# hidden_layer_sizes: 은닉층의 노드 수를 튜플 형태로 지정 (여기서는 두 층, 각각 128와 64 노드)
# max_iter: 최대 반복 횟수, verbose: 학습 과정 출력 여부
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42, verbose=True)

# 학습 데이터(X_train, y_train)를 사용하여 모델을 학습
clf.fit(X_train, y_train)

# 학습된 모델을 사용하여 테스트 데이터(X_test)에 대해 예측을 수행
y_pred = clf.predict(X_test)

# 분류 결과에 대한 자세한 리포트를 출력
# 각 클래스별 정밀도, 재현율, F1 점수 출력
print(classification_report(y_test, y_pred))

# 결과를 저장할 폴더를 생성
os.makedirs("./MLPClassifier/results", exist_ok=True)

# 테스트 정답(y_test)과 예측 결과(y_pred)를 numpy 배열 파일로 저장
# 이 파일들은 후속 분석이나 시각화에 사용할 수 있습니다.
np.save("./MLPClassifier/results/y_test.npy", y_test)
np.save("./MLPClassifier/results/y_pred.npy", y_pred)
print("y_test.npy, y_pred.npy 저장 완료 (MLPClassifier/results/ 폴더)")

# 학습된 MLPClassifier 모델을 joblib을 이용하여 저장
joblib.dump(clf, "./MLPClassifier/results/mlp_model.pkl")
print("MLP 모델 저장 완료 (MLPClassifier/results/mlp_model.pkl)")
