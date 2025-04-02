import cv2
import numpy as np
import joblib
from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont

# 감정 라벨 매핑
emotion_kor_map = {"angry": "짜증", "happy": "행복", "neutral": "무표정", "sad": "슬픔"}
# DeepFace 임베딩 추출에 사용할 모델명
embedding_model_name = "ArcFace"

# 한글 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
font = ImageFont.truetype(font_path, 32)

# 감정 안정화 관련 변수
previous_emotion = ""
emotion_counter = 0
stable_emotion = ""
stable_confidence = 0.0
threshold_frames = 2  # 감정이 연속 프레임에서 유지되어야 업데이트

def load_emotion_model(model_path):
    model = joblib.load(model_path)
    print("학습된 모델 불러오기 완료:", model_path)
    return model

def analyze_emotion(frame, model):

    global previous_emotion, emotion_counter, stable_emotion, stable_confidence
    try:
        # DeepFace로 임베딩 추출 (enforce_detection=True: 얼굴 검출 실패 시 에러 발생)
        representation = DeepFace.represent(img_path=frame,
                                            model_name=embedding_model_name,
                                            enforce_detection=True)
        embedding = representation[0]['embedding']

        # MLP 모델을 사용하여 감정 예측
        pred = model.predict([embedding])[0]
        proba = model.predict_proba([embedding])[0]
        # 해당 감정의 확률 (백분율 계산)
        prob = proba[model.classes_ == pred][0]

        # 감정 안정화 로직: 연속 프레임에서 같은 결과이면 카운트 증가
        if pred == previous_emotion:
            emotion_counter += 1
        else:
            emotion_counter = 0
            previous_emotion = pred

        if emotion_counter >= threshold_frames:
            stable_emotion = pred
            stable_confidence = prob

        if stable_emotion:
            pred_text = emotion_kor_map.get(stable_emotion, stable_emotion)
            return f"감정: {pred_text} ({stable_confidence * 100:.1f}%)"
        else:
            return "감정: 분석 중..."
    except Exception as e:
        print("예측 오류:", e)
        return "감정: 알 수 없음"

def run_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임의 감정 분석 결과 문자열 얻기
        emotion_text = analyze_emotion(frame, model)

        # OpenCV 이미지(BGR)를 Pillow 이미지(RGB)로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # PIL 이미지에 한글 텍스트 그리기
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), emotion_text, font=font, fill=(255, 0, 0))

        # Pillow 이미지를 다시 OpenCV 이미지로 변환 (RGB -> BGR)
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.imshow("Webcam Emotion Recognition", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    default_model_path = "./MLPClassifier/results/mlp_model.pkl"
    model = load_emotion_model(default_model_path)
    run_webcam(model)
