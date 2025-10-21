#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp

# =========================================
# 🧩 Mediapipe 초기화 (Face Detection)
# =========================================
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,            # 0: 가까운 거리, 1: 먼 거리(대상 크기/해상도에 따라 선택)
    min_detection_confidence=0.5  # 탐지 신뢰도
)

# =========================================
# 📸 카메라 / 동영상 연결
# =========================================
# 기본 카메라 사용:
#cap = cv2.VideoCapture(0)

# 동영상 파일 사용(테스트용) — 파일명 바꿔서 사용:
cap = cv2.VideoCapture("face.mp4")

print("📷 비디오 스트림 시작 — ESC를 눌러 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("⚠️ 프레임을 읽지 못했습니다. (스트림/파일 끝) 종료합니다.")
        break

    # 좌우 반전 (셀카 뷰) — 필요 없으면 제거 가능
    image = cv2.flip(image, 1)

    # BGR → RGB 변환 (MediaPipe는 RGB 입력)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 검출 수행
    results = face_detection.process(image_rgb)

    # 얼굴 검출 결과 그리기
    if results.detections:
        for detection in results.detections:
            # MediaPipe의 유틸로 기본 바운딩박스/키포인트를 그림
            mp_drawing.draw_detection(image, detection)

            # 추가: 신뢰도(퍼센트) 텍스트 표시
            if detection.score:
                score = int(detection.score[0] * 100)
                # 바운딩 박스 좌표(픽셀) 계산
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h) - 10
                if y_min < 0:
                    y_min = 0
                # 텍스트 표시
                cv2.putText(
                    image,
                    f"{score}%",
                    (x_min, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # 화면에 표시
    cv2.imshow('😊 MediaPipe Face Detector', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        print("👋 종료합니다.")
        break

# =========================================
# 🔚 종료 처리
# =========================================
cap.release()
cv2.destroyAllWindows()
