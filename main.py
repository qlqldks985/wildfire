import cv2
from ultralytics import YOLO
import time

# ====================== 설정 ======================
# 모델 경로 (현재 당신 프로젝트 구조에 맞춤)
MODEL_PATH = 'runs/segment/fire_smoke-3/weights/best.pt' 

ESP32_URL = "http://192.168.137.72:81/stream"
CONFIDENCE_THRESHOLD = 0.4
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
    print("   3. 브라우저에서 http://192.168.137.72:81/stream 열리는지 테스트")
    exit()

print("✅ ESP32 카메라 연결 성공!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임 읽기 실패... 재시도")
        time.sleep(1)
        continue

    # YOLO 추론
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # 결과 화면에 그리기
    annotated_frame = results[0].plot()

    # FPS 표시
    fps = int(1 / (time.time() - start_time)) if 'start_time' in locals() else 0
    cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    start_time = time.time()

    # 창 표시
    cv2.imshow("Fire & Smoke Detection - ESP32", annotated_frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 정리
cap.release()
cv2.destroyAllWindows()
print("프로그램 종료")