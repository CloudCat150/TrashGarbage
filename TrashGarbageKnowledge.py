import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 그림자 제거 함수
def remove_shadow(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be found.")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    shadow_removed = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        205, 
        15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(shadow_removed, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned

# SIFT 특징 추출 함수
def extract_sift_features(image_path):
    image = remove_shadow(image_path)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# 학습 데이터 준비
def prepare_training_data(image_paths, labels):
    descriptors_list = []
    for image_path in image_paths:
        descriptors = extract_sift_features(image_path)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list, labels

# 특징 벡터로부터 하나의 벡터로 통합
def concatenate_descriptors(descriptor_list):
    concatenated = []
    for descriptors in descriptor_list:
        if descriptors is not None:
            concatenated.append(descriptors.flatten())
    return np.array(concatenated)

# 학습 및 평가
def train_and_evaluate(train_image_paths, train_labels, test_image_path, test_label):
    # 학습 데이터 준비
    train_descriptors_list, train_labels = prepare_training_data(train_image_paths, train_labels)
    X_train = concatenate_descriptors(train_descriptors_list)
    y_train = np.array(train_labels)

    # SVM 모델 훈련
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # 단일 테스트 이미지에 대한 특징 추출 및 예측
    test_descriptors = extract_sift_features(test_image_path)
    if test_descriptors is not None:
        X_test = test_descriptors.flatten().reshape(1, -1)
        y_test = np.array([test_label])

        # 예측
        y_pred = svm.predict(X_test)
        
        print("Prediction for the test image:", y_pred[0])
        print("True label for the test image:", test_label)
    else:
        print("Test image could not be processed.")

# "\Preprocessed Data\can" 폴더에서 이미지 불러오기
def load_images_from_folder(folder_path):
    image_paths = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if img_path.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 확장자 확인
            image_paths.append(img_path)
    return image_paths

# 학습 및 테스트 데이터 폴더 경로
train_folder_path = r"C:\Users\user\Desktop\TrashGarbage\Preprocessed Data\can"  # 학습 데이터 폴더
test_image_path = r"C:\Users\user\Desktop\TrashGarbage\20241216_110613.jpg"  # 단일 테스트 이미지 경로

# 학습 데이터 로딩
train_image_paths = load_images_from_folder(train_folder_path)
train_labels = [0] * len(train_image_paths)  # 모든 학습 이미지를 같은 레이블로 설정 (예: 0)

# 테스트 데이터 레이블 (테스트 이미지 하나에 대한 레이블)
test_label = 0  # 테스트 이미지에 해당하는 레이블 (예: 0)

# 학습 및 평가 실행
train_and_evaluate(train_image_paths, train_labels, test_image_path, test_label)
