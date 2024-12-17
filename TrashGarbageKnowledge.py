import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

# 이미지 전처리 함수
def preprocess_image(img_path, size=(256, 256)):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(binary, size)
    return resized / 255.0  # Normalize to 0-1

# CNN 모델을 사용해 특징 벡터 추출
def get_feature_vector(image):
    model = tf.keras.applications.ResNet50(include_top=False, pooling='avg', input_shape=(256, 256, 3))
    image = np.stack((image,)*3, axis=-1)  # 1채널 -> 3채널
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features.flatten()

def calculate_similarity(new_image_path, database_images):
    new_image = preprocess_image(new_image_path)
    new_vector = get_feature_vector(new_image)
    
    similarities = []
    for img_path in database_images:
        img = preprocess_image(img_path)
        vector = get_feature_vector(img)
        similarity = cosine_similarity([new_vector], [vector])
        similarities.append(similarity[0][0])

    return similarities

database_images = ['img1.png', 'img2.png', ..., 'img300.png']  # 기존 이미지 경로
new_image_path = 'new_image.png'

similarities = calculate_similarity(new_image_path, database_images)

# 결과 출력
for i, sim in enumerate(similarities):
    print(f"Image {i+1}: {sim:.2f}% similarity")
