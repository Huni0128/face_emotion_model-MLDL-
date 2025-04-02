import os
from view import webcam_view

def main():
    model_path = os.path.join("MLPClassifier", "results", "mlp_model.pkl")
    model = webcam_view.load_emotion_model(model_path)
    webcam_view.run_webcam(model)

if __name__ == "__main__":
    main()
