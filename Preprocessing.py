import cv2
import numpy as np
import os

def remove_shadow(image_path, output_path=None):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image at {image_path} could not be found.")
        
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러링으로 노이즈 제거
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 그림자 제거를 위한 Adaptive Threshold 적용
    shadow_removed = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        205, 
        15
    )

    # Morphological operations으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(shadow_removed, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # 결과 저장 (선택)
    if output_path:
        cv2.imwrite(output_path, cleaned)

    return cleaned

# 디렉터리를 처리하는 함수 추가
def process_directory(input_dir, output_dir):
    # 입력 디렉토리 내 모든 파일 확인
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_path):
            try:
                print(f"Processing {input_path}...")
                remove_shadow(input_path, output_path)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

# 사용자로부터 파일명 입력 받기
file_name = input("Directory Name : ").strip()

# 현재 스크립트의 디렉터리
current_dir = os.path.dirname(os.path.abspath(__file__))

# 입력 디렉터리와 출력 디렉터리 경로 설정
input_dir = os.path.join(current_dir, 'Study Data', file_name)
output_dir = os.path.join(current_dir, 'Preprocessed Data', file_name)

# 입력한 파일명에 해당하는 디렉터리 내 모든 이미지 처리
process_directory(input_dir, output_dir)

print(f"All images from '{file_name}' processed.")
