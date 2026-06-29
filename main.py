import cv2
from ultralytics import YOLO
import time

# ====================== 설정 ======================
# 모델 경로 (현재 당신 프로젝트 구조에 맞춤)
# main.py 상단 수정 예시
MODEL_PATH = 'runs/detect/train/weights/best.pt' 

ESP32_URL = "http://192.168.137.178:81/stream"
CONFIDENCE_THRESHOLD = 0.20

# ☀️ 렌즈 밝기 및 대비 설정 (수정 가능한 구역)
# BRIGHTNESS: -255 ~ 255 (음수면 어두워지고, 양수면 밝아집니다)
# CONTRAST: 0.5 ~ 3.0 (1.0이 기본값, 높일수록 명암 대비가 강해집니다)
BRIGHTNESS = 0    # 💡 팁: 창문 빛 번짐을 줄이려면 약간 음수(-20 ~ -50)로 낮춰보세요!
CONTRAST = 1.0      # 💡 팁: 대비를 살짝 높이면 연기와 빛의 경계가 뚜렷해집니다.
# =================================================
# =================================================

print("🔥 Fire & Smoke Detection 시작...")

# 모델 로드
print("📦 모델 로드 중...")
model = YOLO(MODEL_PATH)
print(f"✅ 모델 로드 완료! ({MODEL_PATH})")

# ESP32 카메라 연결
print("📷 ESP32 카메라 연결 시도...")
cap = cv2.VideoCapture(ESP32_URL)

if not cap.isOpened():
    print("❌ ESP32 연결 실패!")
    print("   1. ESP32가 WiFi에 잘 연결되어 있는지")
    print("   2. IP 주소 확인 (192.168.137.72)")
    print("   3. 브라우저에서 http://192.168.137.178:81/stream 열리는지 테스트")
    exit()

print("✅ ESP32 카메라 연결 성공!")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임 읽기 실패... 재시도")
        time.sleep(1)
        continue

    adjusted_frame = cv2.convertScaleAbs(frame, alpha=CONTRAST, beta=BRIGHTNESS)

    # YOLO 추론
    results = model(adjusted_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # 결과 화면에 그리기
    annotated_frame = results[0].plot()

    # FPS 표시
    fps = int(1 / (time.time() - start_time)) if 'start_time' in locals() else 0
    cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    start_time = time.time()

    # 창 표시
    cv2.imshow("Fire & Smoke Detection - ESP32", annotated_frame)

    # 입력된 키보드 값 감지
    key = cv2.waitKey(1) & 0xFF

    # 📸 [단축키 S] 누르면 현재 프레임 캡처 저장
    if key == ord('s'):
        import os
        # 캡처본을 저장할 폴더 생성
        os.makedirs("captures", exist_ok=True)
        
        # 파일명을 현재 시간으로 설정 (예: capture_20260626_123000.jpg)
        filename = f"captures/capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # 원본 이미지(frame) 또는 박스가 그려진 이미지(annotated_frame) 선택해서 저장
        cv2.imwrite(filename, annotated_frame) 
        print(f"📸 사진이 저장되었습니다: {filename}")

    # 'q' 키로 종료
    if key == ord('q'):
        break

# 종료 정리
cap.release()
cv2.destroyAllWindows()
print("프로그램 종료")