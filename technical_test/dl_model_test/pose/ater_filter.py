# person 감지(vest도 감지 하면 허가된 인원) -> crop후 -> opencv로 보조로 형광픽셀 많이 들어있으지 확인.







# YOLO로 감지된 person box crop 후
cropped = frame[y1:y2, x1:x2]
hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

# 형광 노랑 필터링
lower = np.array([25, 150, 150])
upper = np.array([40, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

# 형광색 비율 계산
fluorescent_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
if fluorescent_ratio > 0.3:
    print("✅ 조끼 착용")
else:
    print("❌ 조끼 미착용")



import cv2
import numpy as np
from ultralytics import YOLO

# 1. 모델 불러오기
model = YOLO("yolov8s-seg.pt")

# 2. 테스트용 이미지 로딩
image = cv2.imread("sample.jpg")

# 3. YOLOv8 세그멘테이션 추론
results = model(image)[0]  # results.masks, results.boxes 존재

# 4. 클래스 이름 매핑
CLASS_NAMES = ['bird', 'fod', 'person', 'animal', 'airplane', 'batcar', 'coordinateboard', 'vest', 'drone']

# 5. HSV 색상 범위 (형광 노랑 예시)
HSV_LOWER = np.array([30, 100, 100])
HSV_UPPER = np.array([70, 255, 255])
VEST_COLOR_RATIO_THRESHOLD = 0.05

for i, mask in enumerate(results.masks.data):
    class_id = int(results.boxes.cls[i].item())
    class_name = CLASS_NAMES[class_id]

    # 6. 사람 클래스만 처리
    if class_name != 'person':
        continue

    # 7. 마스크 → numpy 변환
    mask_np = mask.cpu().numpy().astype(np.uint8) * 255

    # 8. 바운딩 박스 추출
    x1, y1, x2, y2 = map(int, results.boxes.xyxy[i].tolist())
    w, h = x2 - x1, y2 - y1

    # 9. 자세 추정 (비율로 standing / lying 구분)
    aspect = h / w
    if aspect > 1.2:
        pose = "standing"
        vest_roi = mask_np[y1:y1+int(h*0.4), x1:x2]
    elif aspect < 0.8:
        pose = "lying"
        vest_roi = mask_np[y1:y2, x1:x2]
    else:
        pose = "ambiguous"
        vest_roi = mask_np[y1:y2, x1:x2]

    # 10. 상체 영역만 HSV 변환
    person_crop = image[y1:y2, x1:x2]
    hsv_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_crop, HSV_LOWER, HSV_UPPER)

    vest_masked = cv2.bitwise_and(hsv_mask, hsv_mask, mask=vest_roi)
    fluo_pixels = np.count_nonzero(vest_masked)
    total_pixels = np.count_nonzero(vest_roi)
    ratio = fluo_pixels / (total_pixels + 1e-5)

    has_vest = ratio > VEST_COLOR_RATIO_THRESHOLD
    vest_status = "Vest" if has_vest else "No Vest"

    # 🔽 출력 구조 (후처리용 구조체처럼 사용 가능)
    output = {
        "class_id": class_id,
        "class_name": class_name,
        "bbox": [x1, y1, x2, y2],
        "pose": pose,
        "vest": "yes" if has_vest else "no",
        "vest_ratio": round(ratio, 3)
    }
    print(f"🧾 객체 {i}: {output}")

    # 11. 시각화
    label = f"{pose} | {vest_status}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0) if has_vest else (0, 0, 255), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

# 12. 전체 결과 보기
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
