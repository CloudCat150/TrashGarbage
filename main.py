from utils import load_trained_model, predict_image_class

def main():
    # 모델 로드
    model = load_trained_model('plastic_trash_classifier.h5')

    # 이미지 경로 (예측할 이미지)
    img_path = 'C:/Users/user/Desktop/image (2).jpg'

    # 예측 수행
    prediction = predict_image_class(model, img_path)
    
    # 예측 결과 출력 (확률이나 클래스 이름 출력)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
