import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# 모델 로드 함수 (학습된 모델 로드)
def load_trained_model(model_path='plastic_trash_classifier.h5'):
    model = load_model(model_path)  # 저장된 모델을 로드
    return model

# 이미지에서 특징 벡터 추출 (이미지를 모델에 맞게 전처리)
def prepare_image(img_path):
    # 이미지 로드, 크기를 128x128로 맞추기
    img = image.load_img(img_path, target_size=(128, 128))  # 모델 입력 크기 맞추기
    img_array = image.img_to_array(img)  # 이미지를 배열로 변환
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 이미지 전처리 (0~1 범위로 정규화)
    return img_array

# 예측 수행 함수
def predict_image_class(model, img_path):
    img_array = prepare_image(img_path)  # 이미지 전처리
    prediction = model.predict(img_array)  # 예측 수행
    return prediction

# 예측된 클래스 해석 함수
def interpret_prediction(prediction, class_indices):
    # 예측된 값에서 가장 높은 확률을 가진 클래스의 인덱스 찾기
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = list(class_indices.keys())[predicted_class_index]
    return predicted_class_name

# 예시 클래스 인덱스 (실제 클래스 인덱스를 여기에 추가)
class_indices = {'plastic': 0, 'metal': 1, 'paper': 2}

# 예측 결과
prediction = [[0.77703047, 0.1957651, 0.02720442]]
predicted_class = interpret_prediction(prediction, class_indices)
print(f"Predicted class: {predicted_class}")
