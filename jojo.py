import cv2
import numpy as np
import mediapipe as mp

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Загружаем изображение стэнда (PNG с прозрачностью)
stand_img = cv2.imread("d4c.png", cv2.IMREAD_UNCHANGED)

# Запуск веб-камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Отзеркалим изображение (чтобы стэнд был справа от тебя)
    frame = cv2.flip(frame, 1)

    # Конвертируем кадр в RGB (MediaPipe требует RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Получаем координаты плеч и бедра
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        # Переводим нормализованные координаты в пиксели
        h, w, _ = frame.shape
        left_x, left_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        right_x, right_y = int(right_shoulder.x * w), int(right_shoulder.y * h)
        hip_x, hip_y = int(left_hip.x * w), int(left_hip.y * h)
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)

        # **Определяем размер стэнда по росту**
        torso_height = abs(hip_y - nose_y)
        full_body_height = abs(hip_y - nose_y) * 2

        if full_body_height > h * 0.8:
            stand_height = full_body_height  # Если стоишь
        else:
            stand_height = int(torso_height * 2)  # Если сидишь

        stand_height = int(stand_height * 0.9)  # Уменьшаем на 10%

        # Масштабируем PNG
        stand_width = int(stand_img.shape[1] * stand_height / stand_img.shape[0])
        stand_resized = cv2.resize(stand_img, (stand_width, stand_height))

        # **Смещаем стэнд дальше вправо**
        stand_x1 = right_x + 130
        stand_y1 = left_y - stand_height // 2 - int(stand_height * 0.1)
        stand_x2 = stand_x1 + stand_width
        stand_y2 = stand_y1 + stand_height

        # Коррекция границ
        stand_x1 = max(0, stand_x1)
        stand_y1 = max(0, stand_y1)
        stand_x2 = min(w, stand_x2)
        stand_y2 = min(h, stand_y2)

        # Вырезаем часть стэнда, если он выходит за границы
        stand_resized = stand_resized[:stand_y2 - stand_y1, :stand_x2 - stand_x1]

        # **Альфа-канал (если PNG)**
        if stand_resized.shape[-1] == 4:
            alpha_s = stand_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # **Накладываем стэнд за человеком без кругов**
            for c in range(3):
                frame[stand_y1:stand_y2, stand_x1:stand_x2, c] = (
                    alpha_s * stand_resized[:, :, c] +
                    alpha_l * frame[stand_y1:stand_y2, stand_x1:stand_x2, c]
                )

    # Показываем результат
    cv2.imshow("JoJo Stand Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
