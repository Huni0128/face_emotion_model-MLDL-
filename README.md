# Face Emotion Recognition Model (MLDL)

얼굴 이미지를 기반으로 감정을 인식하는 시스템입니다.  
DeepFace 라이브러리를 사용해 얼굴 임베딩을 추출하고 scikit-learn의 MLPClassifier를 활용하여 감정을 예측합니다.  
웹캠 영상을 실시간으로 처리하여 분석 결과를 화면에 오버레이하는 기능을 제공합니다.

---

## 프로젝트 개요

- **목표:**  
  얼굴 이미지를 입력받아 DeepFace를 통해 임베딩을 추출한 후, MLPClassifier 모델을 이용하여 감정을 예측합니다.

- **주요 기능:**  
  - DeepFace를 이용한 얼굴 임베딩 추출  
  - 학습된 MLPClassifier 모델로 감정 예측  
  - 웹캠 영상을 통한 실시간 감정 인식  
  - 한글 감정 라벨(예: "짜증", "행복", "무표정", "슬픔") 표시 및 안정화 처리

---

## 파일 및 디렉토리 구조
```bash
face_emotion_model-MLDL-/
├── MLPClassifier/
│   └── results/
│       └── mlp_model.pkl
├── images/
├── train_src/
├── view/
│   └── webcam_view.py
├── .gitignore
├── main.py
└── README.md
```

---

## 사용 기술
| **카테고리**           | **기술 스택**                                                                                                                                                                                                                                                                                         |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS & Editor**        | [![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-E95420?style=flat-square&logo=Ubuntu&logoColor=white)](https://ubuntu.com/) [![VSCode](https://img.shields.io/badge/Editor-VSCode-007ACC?style=flat-square&logo=VisualStudioCode&logoColor=white)](https://code.visualstudio.com/)  |
| **Language & Library** | [![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=NumPy&logoColor=white)](https://numpy.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) <br> [![DeepFace](https://img.shields.io/badge/DeepFace-3C3C3C?style=flat-square)](https://github.com/serengil/deepface) [![Pillow](https://img.shields.io/badge/Pillow-ED5C5C?style=flat-square&logo=Pillow&logoColor=white)](https://python-pillow.org/) [![joblib](https://img.shields.io/badge/joblib-4CAF50?style=flat-square)](https://joblib.readthedocs.io/) |
| **Version Control**    | [![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=Git&logoColor=white)](https://git-scm.com/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white)](https://github.com/)                                                                                       |
                                      |

