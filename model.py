import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_model(input_shape, num_classes):
    model = Sequential([
        # Feature Extraction Layers
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 클래스 수에 맞게 조정
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(data_dir, batch_size=32, epochs=30):
    # 데이터 경로 설정
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 이미지 데이터 증강 및 로드
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),  # 이미지 크기 조정
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # 클래스 수 자동 감지
    num_classes = len(train_gen.class_indices)

    # 모델 생성
    input_shape = (128, 128, 3)  # 이미지 크기와 채널 수
    model = create_model(input_shape, num_classes)

    # 모델 학습
    history = model.fit(
        train_gen,  
        validation_data=val_gen,
        epochs=epochs
    )

    # 모델 저장
    model.save('plastic_trash_classifier.h5')
    print("Model training completed and saved as 'plastic_trash_classifier.h5'.")

if __name__ == "__main__":
    # 사용자로부터 데이터 경로 입력
    data_dir = input("Enter the path to the preprocessed data directory: ").strip()
    train_model(data_dir)
