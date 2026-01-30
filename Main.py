import cv2
import numpy as np
import os


def create_stereo_calibration_images(num_pairs=20):


    os.makedirs("stereo/left", exist_ok=True)
    os.makedirs("stereo/right", exist_ok=True)


    pattern_size = (9, 6)  # 9x6
    square_size = 40
    img_size = (640, 480)
    baseline = 30

    print(f"Создание {num_pairs} стереопар...")

    for i in range(num_pairs):
        for camera_idx, camera in enumerate(['left', 'right']):
            img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 180

            offset_x = -baseline // 2 if camera == 'left' else baseline // 2
            offset_y = np.random.randint(-10, 10)  # небольшое случайное смещение по Y

            for y in range(pattern_size[1] + 1):
                for x in range(pattern_size[0] + 1):
                    color = (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)

                    x_base = img_size[0] // 2 - pattern_size[0] * square_size // 2 + x * square_size
                    y_base = img_size[1] // 2 - pattern_size[1] * square_size // 2 + y * square_size

                    if i > 5:
                        perspective_x = np.random.randint(-20, 20) * (x - pattern_size[0] // 2)
                        perspective_y = np.random.randint(-10, 10) * (y - pattern_size[1] // 2)
                    else:
                        perspective_x = perspective_y = 0

                    x1 = int(x_base + offset_x + perspective_x)
                    y1 = int(y_base + offset_y + perspective_y)
                    x2 = x1 + square_size
                    y2 = y1 + square_size

                    if 0 <= x1 < img_size[0] and 0 <= y1 < img_size[1]:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            if i % 3 == 0:
                img = cv2.GaussianBlur(img, (3, 3), 0.5)

            noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

            filename = f"stereo/{camera}/{camera}_{i:03d}.jpg"
            cv2.imwrite(filename, img)

        print(f"  Создана стереопара {i + 1}/{num_pairs}")

    print(f"\n Создано {num_pairs} стереопар в папках stereo/left/ и stereo/right/")

    create_test_objects()


def create_test_objects():


    print("\nСоздание тестовых объектов для измерений...")

    for camera in ['left', 'right']:
        img = np.ones((480, 640, 3), dtype=np.uint8) * 150  # Серый фон

        offset_x = -15 if camera == 'left' else 15

        center_x, center_y = 320 + offset_x, 240
        cv2.ellipse(img, (center_x, center_y + 40), (60, 30),
                    0, 0, 360, (100, 100, 100), -1)

        cv2.ellipse(img, (center_x, center_y), (50, 70),
                    0, 0, 360, (180, 180, 180), -1)
        cv2.ellipse(img, (center_x, center_y), (50, 70),
                    0, 0, 360, (80, 80, 80), 2)

        cv2.ellipse(img, (center_x, center_y - 70), (55, 20),
                    0, 0, 360, (220, 220, 220), -1)
        cv2.ellipse(img, (center_x, center_y - 70), (55, 20),
                    0, 0, 360, (60, 60, 60), 2)

        if camera == 'left':
            cv2.ellipse(img, (center_x - 70, center_y - 20), (30, 15),
                        90, 0, 360, (120, 120, 120), -1)
        else:
            cv2.ellipse(img, (center_x + 70, center_y - 20), (30, 15),
                        -90, 0, 360, (120, 120, 120), -1)

        cv2.ellipse(img, (center_x + 10, center_y + 50), (65, 35),
                    0, 0, 360, (80, 80, 80), -1)
        cv2.ellipse(img, (center_x + 10, center_y + 50), (65, 35),
                    0, 0, 360, (80, 80, 80), -1)

        cv2.putText(img, f"Cup - {camera} view", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite(f"stereo/test_{camera}.jpg", img)
        print(f"  Создано: stereo/test_{camera}.jpg")

    for camera in ['left', 'right']:
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200

        offset_x = -10 if camera == 'left' else 10

        cube_size = 80
        x, y = 320 + offset_x - cube_size // 2, 240 - cube_size // 2

        cv2.rectangle(img, (x, y), (x + cube_size, y + cube_size), (220, 220, 220), -1)
        cv2.rectangle(img, (x, y), (x + cube_size, y + cube_size), (100, 100, 100), 2)

        pts = np.array([[x, y], [x + cube_size, y],
                        [x + cube_size - 20, y - 20], [x - 20, y - 20]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (200, 200, 200))

        pts = np.array([[x + cube_size, y], [x + cube_size, y + cube_size],
                        [x + cube_size - 20, y + cube_size - 20], [x + cube_size - 20, y - 20]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (180, 180, 180))

        cv2.putText(img, f"Uniform - {camera}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imwrite(f"stereo/uniform_{camera}.jpg", img)
        print(f"  Создано: stereo/uniform_{camera}.jpg")

    for camera in ['left', 'right']:
        img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (25, 25), 0)

        offset_x = -10 if camera == 'left' else 10

        center = (320 + offset_x, 240)
        radius = 60

        for r in range(radius, 0, -1):
            intensity = int(200 - (r / radius) * 80)
            cv2.circle(img, center, r, (intensity, intensity, intensity), -1)

        cv2.circle(img, (center[0] - 15, center[1] - 15), 12, (240, 240, 240), -1)
        cv2.circle(img, center, radius, (60, 60, 60), 2)

        cv2.ellipse(img, (center[0] + 15, center[1] + radius + 10),
                    (radius, radius // 3), 0, 0, 360, (80, 80, 80), -1)

        cv2.putText(img, f"Textured - {camera}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite(f"stereo/textured_{camera}.jpg", img)
        print(f"  Создано: stereo/textured_{camera}.jpg")

    print("\n✓ Все тестовые изображения созданы!")


if __name__ == "__main__":
    create_stereo_calibration_images(20)
    print("\nГотово к запуску Task 2!")


