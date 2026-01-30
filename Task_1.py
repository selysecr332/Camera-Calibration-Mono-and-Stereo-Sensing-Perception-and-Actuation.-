
"""
Task 1: Калибровка монокамеры
"""

import cv2
import numpy as np
import glob
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.optimize import least_squares
from typing import List, Tuple, Dict, Any


class MonoCameraCalibrator:

    def __init__(self, chessboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):

        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.calibrated = False

        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

        self.objpoints = []
        self.imgpoints = []
        self.calibration_images = []

        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None

    def load_images(self, image_paths: List[str],
                    show_corners: bool = True) -> int:

        self.calibration_images = []
        self.objpoints = []
        self.imgpoints = []

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for i, fname in enumerate(image_paths):
            print(f"Обработка изображения {i + 1}/{len(image_paths)}: {fname}")

            img = cv2.imread(fname)
            if img is None:
                print(f"  Ошибка загрузки: {fname}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

                if show_corners:
                    img_display = cv2.drawChessboardCorners(img.copy(),
                                                            self.chessboard_size,
                                                            corners2, ret)
                    cv2.imshow('Найденные углы', img_display)
                    cv2.waitKey(500)

                self.calibration_images.append(img)
            else:
                print(f"  Углы не найдены: {fname}")

        cv2.destroyAllWindows()
        return len(self.objpoints)

    def calibrate(self) -> float:

        if len(self.objpoints) < 10:
            raise ValueError(f"Недостаточно изображений для калибровки. Найдено: {len(self.objpoints)}")

        print(f"Начинаем калибровку с {len(self.objpoints)} изображений...")

        img_size = self.calibration_images[0].shape[1], self.calibration_images[0].shape[0]

        ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None
        )

        self.reprojection_error = self._calculate_reprojection_error()

        self.calibrated = True
        print(f"Калибровка завершена. Ошибка репроекции: {self.reprojection_error:.4f} пикселей")

        return self.reprojection_error

    def _calculate_reprojection_error(self) -> float:
        total_error = 0
        total_points = 0

        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i],
                                              self.rvecs[i],
                                              self.tvecs[i],
                                              self.mtx,
                                              self.dist)

            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error * len(imgpoints2)
            total_points += len(imgpoints2)

        return total_error / total_points

    def save_calibration(self, filename: str = "mono_calibration.json"):
        if not self.calibrated:
            raise ValueError("Камера не откалибрована")

        calibration_data = {
            "date": datetime.now().isoformat(),
            "chessboard_size": self.chessboard_size,
            "square_size": self.square_size,
            "camera_matrix": self.mtx.tolist(),
            "distortion_coefficients": self.dist.tolist(),
            "reprojection_error": float(self.reprojection_error),
            "image_count": len(self.objpoints)
        }

        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)

        print(f"Калибровка сохранена в {filename}")

    def load_calibration(self, filename: str = "mono_calibration.json"):
        with open(filename, 'r') as f:
            calibration_data = json.load(f)

        self.mtx = np.array(calibration_data["camera_matrix"])
        self.dist = np.array(calibration_data["distortion_coefficients"])
        self.reprojection_error = calibration_data["reprojection_error"]
        self.calibrated = True

        print(f"Калибровка загружена из {filename}")

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        if not self.calibrated:
            raise ValueError("Камера не откалибрована")

        h, w = img.shape[:2]

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))

        undistorted = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)

        return undistorted

    def measure_object(self, img_path: str, real_size_mm: Tuple[float, float] = None):

        if not self.calibrated:
            raise ValueError("Камера не откалибрована")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        img_undistorted = self.undistort_image(img)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
        plt.title("Кликните на 4 угла объекта (по часовой стрелке)")
        plt.axis('on')

        points = plt.ginput(4, timeout=0)
        plt.close()

        if len(points) != 4:
            raise ValueError("Необходимо выбрать 4 точки")

        img_points = np.array(points, dtype=np.float32)

        if real_size_mm is None:
            object_width_mm = 25.0
            object_height_mm = 25.0
        else:
            object_width_mm, object_height_mm = real_size_mm

        obj_points = np.array([
            [0, 0, 0],
            [object_width_mm, 0, 0],
            [object_width_mm, object_height_mm, 0],
            [0, object_height_mm, 0]
        ], dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.mtx, self.dist)

        if not ret:
            raise ValueError("Не удалось решить PnP задачу")


        projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, self.mtx, self.dist)

        distance = np.linalg.norm(tvec) / 10.0

        marker_size_mm = 25.0

        pixel_per_mm_x = (img_points[1, 0] - img_points[0, 0]) / object_width_mm
        pixel_per_mm_y = (img_points[2, 1] - img_points[0, 1]) / object_height_mm
        pixel_per_mm = (pixel_per_mm_x + pixel_per_mm_y) / 2.0

        results = {
            "image_size": img.shape[:2],
            "selected_points": img_points.tolist(),
            "projected_points": projected_points.reshape(-1, 2).tolist(),
            "camera_position": tvec.flatten().tolist(),
            "camera_rotation": rvec.flatten().tolist(),
            "distance_to_object_cm": distance,
            "pixels_per_mm": pixel_per_mm,
            "measurement_accuracy": self._calculate_measurement_accuracy(img_points, projected_points)
        }


        self._visualize_measurement(img_undistorted, img_points, projected_points, results)

        return results

    def _calculate_measurement_accuracy(self, img_points, projected_points):
        error = np.mean(np.sqrt(np.sum((img_points - projected_points.reshape(-1, 2)) ** 2, axis=1)))
        return error

    def _visualize_measurement(self, img, img_points, projected_points, results):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))


        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].scatter(img_points[:, 0], img_points[:, 1], c='red', s=100, marker='o', label='Выбранные точки')
        axes[0].plot(np.append(img_points[:, 0], img_points[0, 0]),
                     np.append(img_points[:, 1], img_points[0, 1]),
                     'g-', linewidth=2)
        axes[0].set_title('Изображение с измеряемым объектом')
        axes[0].axis('off')
        axes[0].legend()

        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].scatter(img_points[:, 0], img_points[:, 1], c='red', s=100, marker='o', label='Выбранные')
        axes[1].scatter(projected_points[:, 0, 0], projected_points[:, 0, 1],
                        c='blue', s=100, marker='x', label='Спроецированные')

        for i in range(4):
            axes[1].plot([img_points[i, 0], projected_points[i, 0, 0]],
                         [img_points[i, 1], projected_points[i, 0, 1]],
                         'y--', alpha=0.5)

        axes[1].set_title('Сравнение точек (ошибка: {:.2f} пикс)'.format(results['measurement_accuracy']))
        axes[1].axis('off')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('measurement_results.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualize_calibration(self):
        if not self.calibrated:
            raise ValueError("Камера не откалибрована")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))


        img_idx = min(5, len(self.calibration_images) - 1)
        img_with_corners = cv2.drawChessboardCorners(
            self.calibration_images[img_idx].copy(),
            self.chessboard_size,
            self.imgpoints[img_idx],
            True
        )
        axes[0, 0].imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Пример обнаружения углов (изобр. {img_idx + 1})')
        axes[0, 0].axis('off')

        img_idx = min(10, len(self.calibration_images) - 1)
        img_original = self.calibration_images[img_idx]
        img_undistorted = self.undistort_image(img_original)

        axes[0, 1].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Исходное изображение')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('После коррекции дисторсии')
        axes[0, 2].axis('off')

        im = axes[1, 0].imshow(self.mtx, cmap='viridis', aspect='auto')
        axes[1, 0].set_title('Матрица камеры K')
        axes[1, 0].set_xlabel('Колонки')
        axes[1, 0].set_ylabel('Строки')
        plt.colorbar(im, ax=axes[1, 0])

        coeff_labels = ['k1', 'k2', 'p1', 'p2', 'k3']
        coeff_values = self.dist.flatten()[:5]

        axes[1, 1].bar(coeff_labels[:len(coeff_values)], coeff_values)
        axes[1, 1].set_title('Коэффициенты дисторсии')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].grid(True, alpha=0.3)

        errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i],
                                              self.rvecs[i],
                                              self.tvecs[i],
                                              self.mtx,
                                              self.dist)
            error = np.mean(np.sqrt(np.sum((self.imgpoints[i] - imgpoints2) ** 2, axis=2)))
            errors.append(error)

        axes[1, 2].plot(range(1, len(errors) + 1), errors, 'b-o', linewidth=2, markersize=6)
        axes[1, 2].axhline(y=self.reprojection_error, color='r', linestyle='--',
                           label=f'Средняя: {self.reprojection_error:.3f}')
        axes[1, 2].set_title('Ошибки репроекции по изображениям')
        axes[1, 2].set_xlabel('Номер изображения')
        axes[1, 2].set_ylabel('Ошибка (пиксели)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.suptitle(f'Результаты калибровки монокамеры\n'
                     f'Изображений: {len(self.objpoints)}, '
                     f'Ошибка: {self.reprojection_error:.4f} пикс.',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('mono_calibration_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def main_mono_calibration():
    print("=" * 60)
    print("Task 1: Калибровка монокамеры")
    print("=" * 60)

    calibrator = MonoCameraCalibrator(chessboard_size=(9, 6), square_size=25.0)

    image_paths = sorted(glob.glob("calibration_images/*.jpg") +
                         glob.glob("calibration_images/*.png"))

    if not image_paths:
        print("ОШИБКА: Не найдены изображения для калибровки")
        print("Поместите изображения в папку 'calibration_images/'")
        return

    num_images = calibrator.load_images(image_paths, show_corners=True)
    print(f"\nУспешно обработано {num_images} изображений")

    if num_images < 10:
        print(f"ВНИМАНИЕ: Рекомендуется минимум 10 изображений для калибровки")

    try:
        error = calibrator.calibrate()
        print(f"\nКалибровка успешна!")
        print(f"Матрица камеры K:\n{calibrator.mtx}")
        print(f"\nКоэффициенты дисторсии:\n{calibrator.dist}")

        calibrator.save_calibration("mono_calibration.json")

        calibrator.visualize_calibration()

        print("\n" + "=" * 60)
        print("Измерение объекта на изображении")
        print("=" * 60)

        measurement_image = "measurement_image.jpg"  # Замените на путь к вашему изображению

        if os.path.exists(measurement_image):
            results = calibrator.measure_object(measurement_image, real_size_mm=(25.0, 25.0))

            print(f"\nРезультаты измерений:")
            print(f"Расстояние до объекта: {results['distance_to_object_cm']:.1f} см")
            print(f"Пикселей на мм: {results['pixels_per_mm']:.2f}")
            print(f"Точность измерения: {results['measurement_accuracy']:.2f} пикселей")

            real_width = 25.0
            real_height = 25.0

            measured_width_px = results['selected_points'][1][0] - results['selected_points'][0][0]
            measured_height_px = results['selected_points'][2][1] - results['selected_points'][0][1]

            measured_width_mm = measured_width_px / results['pixels_per_mm']
            measured_height_mm = measured_height_px / results['pixels_per_mm']

            print(f"\nСравнение размеров:")
            print(f"Ширина: реальная={real_width} мм, измеренная={measured_width_mm:.1f} мм, "
                  f"ошибка={abs(real_width - measured_width_mm):.1f} мм ({abs(real_width - measured_width_mm) / real_width * 100:.1f}%)")
            print(f"Высота: реальная={real_height} мм, измеренная={measured_height_mm:.1f} мм, "
                  f"ошибка={abs(real_height - measured_height_mm):.1f} мм ({abs(real_height - measured_height_mm) / real_height * 100:.1f}%)")

        else:
            print(f"Изображение для измерений не найдено: {measurement_image}")
            print("Создайте тестовое изображение с объектом известного размера")

    except Exception as e:
        print(f"Ошибка при калибровке: {e}")

    print("\n" + "=" * 60)
    print("Task 1 завершена!")
    print("=" * 60)


if __name__ == "__main__":
    main_mono_calibration()
"""
source /home/kali/Sensors_HomeWork/pythonProject/.venv/bin/activate
"""