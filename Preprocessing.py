import cv2
import numpy as np
import os

def remove_shadow(image_path, output_path=None):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be found.")
        
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러링 처리로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Threshold로 그림자 제거
    shadow_removed = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        205, 
        15
    )

    # Morphological 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(shadow_removed, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 결과 이미지 저장 (선택 사항)
    if output_path:
        cv2.imwrite(output_path, cleaned)

    return cleaned

def process_directory(input_dir, output_dir):
    """
    입력 디렉토리의 모든 이미지를 전처리하여 출력 디렉토리에 저장.
    """
    # 출력 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 디렉토리 내 파일 순회
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_path):
            try:
                print(f"Processing: {input_path}")
                remove_shadow(input_path, output_path)
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

def preprocess_all_images():
    """
    사용자 입력을 받아 전체 이미지 데이터를 처리하는 함수.
    """
    # 사용자로부터 입력 디렉토리명 받기
    file_name = input("Directory Name : ").strip()

    # 입력 및 출력 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'Study Data', file_name)
    output_dir = os.path.join(current_dir, 'Preprocessed Data', file_name)

    # 전처리 수행
    if os.path.exists(input_dir):
        process_directory(input_dir, output_dir)
        print(f"All images from '{file_name}' have been preprocessed successfully.")
    else:
        print(f"Error: Directory '{input_dir}' does not exist.")

if __name__ == "__main__":
    preprocess_all_images()
