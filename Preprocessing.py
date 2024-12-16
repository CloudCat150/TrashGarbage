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

# 사용 예시
# 현재 스크립트의 디렉터리
current_dir = os.path.dirname(os.path.abspath(__file__))

# 입력 파일과 출력 파일 경로
input_image = os.path.join(current_dir, 'Study Data', 'can', '1.jpg')
output_image = os.path.join(current_dir, 'Preprocessed Data', 'can', '1.jpg')

# 경로 확인
print(f"Input Image Path: {input_image}")
print(f"Output Image Path: {output_image}")

result = remove_shadow(input_image, output_image)

# 결과 시각화 (테스트용)
cv2.imshow("Preprocessed Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
