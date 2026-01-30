
"""
Task 3: Depth Anything - сравнение нейросетевой оценки глубины со стереопарой
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import os
import json
from scipy import ndimage
import time
from typing import Tuple, Dict, List, Optional


class DepthAnythingComparator:

    def __init__(self, stereo_calibration_path: str = "stereo_calibration.json",
                 model_type: str = "small"):
        """
        Инициализация компаратора

        Args:
            stereo_calibration_path: Путь к калибровке стереопары
            model_type: Тип модели Depth Anything ('small', 'base', 'large')
        """
        self.stereo_calibration_path = stereo_calibration_path
        self.model_type = model_type

        # Загрузка стереокалибровки
        self.stereo_params = None
        self.load_stereo_calibration()

        # Инициализация Depth Anything
        self.depth_anything_model = None
        self.transform = None
        self.device = None
        self.initialize_depth_anything()

        # Метрики для сравнения
        self.comparison_results = {}

    def load_stereo_calibration(self):
        """Загрузка параметров стереокалибровки"""
        try:
            with open(self.stereo_calibration_path, 'r') as f:
                self.stereo_params = json.load(f)
            print(f"Стереокалибровка загружена из {self.stereo_calibration_path}")
        except Exception as e:
            print(f"Ошибка загрузки стереокалибровки: {e}")
            self.stereo_params = None

    def initialize_depth_anything(self):
        """Инициализация модели Depth Anything"""
        print(f"Инициализация Depth Anything (модель: {self.model_type})...")

        # Определение устройства
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {self.device}")

        try:
            # Импорт необходимых модулей
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            # Загрузка модели и процессора
            if self.model_type == "small":
                model_name = "depth-anything/Depth-Anything-V2-Small-hf"
            elif self.model_type == "base":
                model_name = "depth-anything/Depth-Anything-V2-Base-hf"
            elif self.model_type == "large":
                model_name = "depth-anything/Depth-Anything-V2-Large-hf"
            else:
                raise ValueError(f"Неизвестный тип модели: {self.model_type}")

            print(f"Загрузка модели {model_name}...")
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.depth_anything_model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)

            # Установка модели в режим оценки
            self.depth_anything_model.eval()

            # Преобразования для изображения
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            print("Depth Anything инициализирована успешно")

        except ImportError as e:
            print(f"Ошибка импорта библиотек для Depth Anything: {e}")
            print("Установите transformers: pip install transformers")
            self.depth_anything_model = None

    def compute_stereo_depth(self, img_left_path: str, img_right_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Вычисление глубины с помощью стереопары

        Args:
            img_left_path: Путь к изображению левой камеры
            img_right_path: Путь к изображению правой камеры

        Returns:
            Карта глубины и метрики качества
        """
        if self.stereo_params is None:
            raise ValueError("Стереокалибровка не загружена")

        # Загрузка изображений
        img_left = cv2.imread(img_left_path)
        img_right = cv2.imread(img_right_path)

        if img_left is None or img_right is None:
            raise ValueError("Не удалось загрузить стереоизображения")

        # Параметры стереопары
        mtx_left = np.array(self.stereo_params["camera_matrix_left"])
        dist_left = np.array(self.stereo_params["distortion_left"])
        mtx_right = np.array(self.stereo_params["camera_matrix_right"])
        dist_right = np.array(self.stereo_params["distortion_right"])
        R = np.array(self.stereo_params["rotation_matrix"])
        T = np.array(self.stereo_params["translation_vector"])

        # Размер изображения
        h, w = img_left.shape[:2]

        # Стереоректификация
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left, dist_left,
            mtx_right, dist_right,
            (w, h), R, T,
            alpha=0
        )

        # Карты преобразования
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, (w, h), cv2.CV_16SC2
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, (w, h), cv2.CV_16SC2
        )

        # Ректификация
        img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

        # Вычисление диспаритета (SGBM)
        window_size = 3
        min_disp = 0
        num_disp = 112 - min_disp

        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        gray_left = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2GRAY)

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Вычисление глубины
        focal_length = mtx_left[0, 0]
        baseline = np.linalg.norm(T)

        # Избегаем деления на ноль
        disparity[disparity == 0] = 0.1

        depth = (focal_length * baseline) / disparity

        # Метрики качества
        metrics = {
            "focal_length": float(focal_length),
            "baseline_mm": float(baseline),
            "image_size": (h, w),
            "disparity_range": (float(np.min(disparity[disparity > 0])),
                                float(np.max(disparity))),
            "depth_range": (float(np.min(depth[depth > 0])),
                            float(np.max(depth))),
            "valid_pixels_ratio": float(np.sum(depth > 0) / (h * w))
        }

        return depth, metrics

    def compute_depth_anything(self, img_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Вычисление глубины с помощью Depth Anything

        Args:
            img_path: Путь к изображению

        Returns:
            Карта глубины и метрики
        """
        if self.depth_anything_model is None:
            raise ValueError("Depth Anything не инициализирована")

        # Загрузка изображения
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (width, height)

        # Подготовка изображения для модели
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Вычисление глубины
        with torch.no_grad():
            outputs = self.depth_anything_model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Интерполяция до оригинального размера
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size[::-1],  # (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Преобразование в numpy
        depth = prediction.squeeze().cpu().numpy()

        # Нормализация и инвертирование (ближе = больше значение)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = 1.0 - depth  # Инвертирование: ближе = меньше значение

        # Масштабирование до метрической системы (примерное)
        # В реальности требуется калибровка для метрической точности
        depth_metric = depth * 5000  # Примерное масштабирование

        metrics = {
            "original_size": original_size,
            "model_input_size": (518, 518),
            "depth_range": (float(depth.min()), float(depth.max())),
            "depth_metric_range": (float(depth_metric.min()), float(depth_metric.max())),
            "model_type": self.model_type
        }

        return depth_metric, metrics

    def compare_depth_maps(self, stereo_depth: np.ndarray,
                           da_depth: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> Dict:
        """
        Сравнение двух карт глубины

        Args:
            stereo_depth: Карта глубины от стереопары
            da_depth: Карта глубины от Depth Anything
            mask: Маска для сравнения (опционально)

        Returns:
            Словарь с метриками сравнения
        """
        # Приведение к одному размеру
        h_stereo, w_stereo = stereo_depth.shape
        h_da, w_da = da_depth.shape

        if (h_stereo, w_stereo) != (h_da, w_da):
            print(f"Размеры не совпадают: стерео={stereo_depth.shape}, DA={da_depth.shape}")
            # Масштабирование Depth Anything до размера стерео
            da_depth_resized = cv2.resize(da_depth, (w_stereo, h_stereo),
                                          interpolation=cv2.INTER_LINEAR)
        else:
            da_depth_resized = da_depth

        # Нормализация для сравнения
        stereo_norm = stereo_depth.copy()
        da_norm = da_depth_resized.copy()

        # Маска валидных пикселей
        if mask is None:
            mask = np.ones_like(stereo_norm, dtype=bool)

        # Игнорируем нулевые и отрицательные значения
        valid_mask = mask & (stereo_norm > 0) & (da_norm > 0)

        if np.sum(valid_mask) == 0:
            print("Нет валидных пикселей для сравнения")
            return {}

        stereo_valid = stereo_norm[valid_mask]
        da_valid = da_norm[valid_mask]

        # Масштабирование Depth Anything к диапазону стерео
        # (поскольку Depth Anything дает относительную глубину)
        if np.std(da_valid) > 0:
            # Линейная регрессия для масштабирования
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                da_valid, stereo_valid
            )
            da_scaled = da_valid * slope + intercept
        else:
            da_scaled = da_valid

        # Метрики сравнения
        metrics = {
            # Абсолютные ошибки
            "mae": float(np.mean(np.abs(stereo_valid - da_scaled))),
            "mse": float(np.mean((stereo_valid - da_scaled) ** 2)),
            "rmse": float(np.sqrt(np.mean((stereo_valid - da_scaled) ** 2))),

            # Относительные ошибки
            "abs_rel": float(np.mean(np.abs(stereo_valid - da_scaled) / stereo_valid)),
            "sq_rel": float(np.mean(((stereo_valid - da_scaled) ** 2) / stereo_valid)),

            # Процентные ошибки
            "delta1": float(np.mean(np.maximum(stereo_valid / da_scaled,
                                               da_scaled / stereo_valid) < 1.25)),
            "delta2": float(np.mean(np.maximum(stereo_valid / da_scaled,
                                               da_scaled / stereo_valid) < 1.25 ** 2)),
            "delta3": float(np.mean(np.maximum(stereo_valid / da_scaled,
                                               da_scaled / stereo_valid) < 1.25 ** 3)),

            # Корреляция
            "pearson_corr": float(np.corrcoef(stereo_valid, da_scaled)[0, 1]),
            "spearman_corr": float(stats.spearmanr(stereo_valid, da_scaled)[0]),

            # Статистика
            "valid_pixels": int(np.sum(valid_mask)),
            "total_pixels": int(np.prod(stereo_norm.shape)),
            "valid_ratio": float(np.sum(valid_mask) / np.prod(stereo_norm.shape)),

            # Диапазоны
            "stereo_range": (float(np.min(stereo_valid)), float(np.max(stereo_valid))),
            "da_range": (float(np.min(da_valid)), float(np.max(da_valid))),
            "da_scaled_range": (float(np.min(da_scaled)), float(np.max(da_scaled))),
        }

        return metrics

    def process_comparison(self, stereo_left_path: str,
                           stereo_right_path: str,
                           da_image_path: str,
                           scene_name: str = "default"):
        """
        Полная обработка сравнения для одной сцены

        Args:
            stereo_left_path: Путь к левому стереоизображению
            stereo_right_path: Путь к правому стереоизображению
            da_image_path: Путь к изображению для Depth Anything
            scene_name: Название сцены
        """
        print(f"\nОбработка сцены: {scene_name}")
        print("-" * 40)

        results = {
            "scene_name": scene_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stereo_images": (stereo_left_path, stereo_right_path),
            "da_image": da_image_path
        }

        try:
            # 1. Вычисление глубины стереопарой
            print("1. Вычисление глубины стереопарой...")
            start_time = time.time()
            stereo_depth, stereo_metrics = self.compute_stereo_depth(
                stereo_left_path, stereo_right_path
            )
            stereo_time = time.time() - start_time
            results["stereo_metrics"] = stereo_metrics
            results["stereo_time_seconds"] = stereo_time

            print(f"   Время: {stereo_time:.2f} сек")
            print(f"   Диапазон глубины: {stereo_metrics['depth_range'][0]:.1f} - "
                  f"{stereo_metrics['depth_range'][1]:.1f} мм")
            print(f"   Валидных пикселей: {stereo_metrics['valid_pixels_ratio'] * 100:.1f}%")

            # 2. Вычисление глубины Depth Anything
            print("2. Вычисление глубины Depth Anything...")
            start_time = time.time()
            da_depth, da_metrics = self.compute_depth_anything(da_image_path)
            da_time = time.time() - start_time
            results["da_metrics"] = da_metrics
            results["da_time_seconds"] = da_time

            print(f"   Время: {da_time:.2f} сек")
            print(f"   Диапазон глубины: {da_metrics['depth_metric_range'][0]:.1f} - "
                  f"{da_metrics['depth_metric_range'][1]:.1f} мм")
            print(f"   Размер модели: {da_metrics['model_input_size']}")

            # 3. Сравнение карт глубины
            print("3. Сравнение карт глубины...")
            comparison_metrics = self.compare_depth_maps(stereo_depth, da_depth)
            results["comparison_metrics"] = comparison_metrics

            if comparison_metrics:
                print(f"   MAE: {comparison_metrics['mae']:.1f} мм")
                print(f"   RMSE: {comparison_metrics['rmse']:.1f} мм")
                print(f"   Относительная ошибка: {comparison_metrics['abs_rel'] * 100:.1f}%")
                print(f"   δ1: {comparison_metrics['delta1'] * 100:.1f}%")
                print(f"   Корреляция: {comparison_metrics['pearson_corr']:.3f}")

            # 4. Визуализация
            print("4. Визуализация результатов...")
            self.visualize_comparison(
                stereo_left_path, stereo_depth, da_depth,
                comparison_metrics, scene_name
            )

            # Сохранение результатов
            self.comparison_results[scene_name] = results

            print(f"\nСцена '{scene_name}' обработана успешно")

        except Exception as e:
            print(f"Ошибка при обработке сцены '{scene_name}': {e}")
            import traceback
            traceback.print_exc()

    def visualize_comparison(self, original_img_path: str,
                             stereo_depth: np.ndarray,
                             da_depth: np.ndarray,
                             metrics: Dict,
                             scene_name: str):
        """
        Визуализация сравнения

        Args:
            original_img_path: Путь к оригинальному изображению
            stereo_depth: Карта глубины от стереопары
            da_depth: Карта глубины от Depth Anything
            metrics: Метрики сравнения
            scene_name: Название сцены
        """
        # Загрузка оригинального изображения
        original_img = cv2.imread(original_img_path)
        if original_img is not None:
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Приведение к одному размеру
        h_stereo, w_stereo = stereo_depth.shape
        da_depth_resized = cv2.resize(da_depth, (w_stereo, h_stereo),
                                      interpolation=cv2.INTER_LINEAR)

        # Создание фигуры
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Оригинальное изображение
        if original_img is not None:
            axes[0, 0].imshow(original_img)
            axes[0, 0].set_title('Оригинальное изображение')
        axes[0, 0].axis('off')

        # 2. Карта глубины от стереопары
        im1 = axes[0, 1].imshow(stereo_depth, cmap='jet')
        axes[0, 1].set_title('Стереопара: карта глубины')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04,
                     label='Глубина (мм)')

        # 3. Карта глубины от Depth Anything
        im2 = axes[0, 2].imshow(da_depth_resized, cmap='jet')
        axes[0, 2].set_title(f'Depth Anything: карта глубины')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04,
                     label='Глубина (мм)')

        # 4. Разностная карта
        diff = np.abs(stereo_depth - da_depth_resized)
        # Маска для валидных значений
        valid_mask = (stereo_depth > 0) & (da_depth_resized > 0)
        diff[~valid_mask] = np.nan

        im3 = axes[1, 0].imshow(diff, cmap='hot')
        axes[1, 0].set_title('Абсолютная разность')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04,
                     label='Разность (мм)')

        # 5. Гистограммы глубины
        axes[1, 1].hist(stereo_depth[stereo_depth > 0].flatten(),
                        bins=50, alpha=0.7, label='Стереопара',
                        color='blue', density=True)
        axes[1, 1].hist(da_depth_resized[da_depth_resized > 0].flatten(),
                        bins=50, alpha=0.7, label='Depth Anything',
                        color='red', density=True)
        axes[1, 1].set_title('Распределение глубины')
        axes[1, 1].set_xlabel('Глубина (мм)')
        axes[1, 1].set_ylabel('Плотность')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Метрики сравнения
        if metrics:
            metrics_text = f"""
            Метрики сравнения:

            Абсолютные ошибки:
              MAE: {metrics.get('mae', 0):.1f} мм
              RMSE: {metrics.get('rmse', 0):.1f} мм

            Относительные ошибки:
              Abs Rel: {metrics.get('abs_rel', 0) * 100:.1f}%
              Sq Rel: {metrics.get('sq_rel', 0) * 100:.1f}%

            Точность (δ):
              δ1: {metrics.get('delta1', 0) * 100:.1f}%
              δ2: {metrics.get('delta2', 0) * 100:.1f}%
              δ3: {metrics.get('delta3', 0) * 100:.1f}%

            Корреляция:
              Pearson: {metrics.get('pearson_corr', 0):.3f}
              Spearman: {metrics.get('spearman_corr', 0):.3f}

            Статистика:
              Валидных пикселей: {metrics.get('valid_pixels', 0):,}
              Процент валидных: {metrics.get('valid_ratio', 0) * 100:.1f}%
            """

            axes[1, 2].text(0.1, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                            verticalalignment='center', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].axis('off')

        plt.suptitle(f'Сравнение оценки глубины: {scene_name}\n'
                     f'Depth Anything ({self.model_type}) vs Стереопара',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        # Сохранение
        filename = f"comparison_{scene_name.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"Визуализация сохранена в {filename}")

    def run_comparison_for_backgrounds(self):
        """Запуск сравнения для разных фонов (как в Task 2)"""
        print("=" * 60)
        print("Сравнение для разных типов фона")
        print("=" * 60)

        # Определение путей к изображениям
        backgrounds = [
            ("Однородный фон",
             "stereo/uniform_left.jpg",
             "stereo/uniform_right.jpg",
             "stereo/uniform_left.jpg"),  # Используем левое изображение для DA

            ("Пестрый фон",
             "stereo/textured_left.jpg",
             "stereo/textured_right.jpg",
             "stereo/textured_left.jpg")
        ]

        for name, left_path, right_path, da_path in backgrounds:
            if (os.path.exists(left_path) and
                    os.path.exists(right_path) and
                    os.path.exists(da_path)):

                self.process_comparison(
                    stereo_left_path=left_path,
                    stereo_right_path=right_path,
                    da_image_path=da_path,
                    scene_name=name
                )
            else:
                print(f"Изображения для {name} не найдены")

        # Сводный отчет
        self.generate_summary_report()

    def generate_summary_report(self):
        """Генерация сводного отчета по всем сравнениям"""
        if not self.comparison_results:
            print("Нет данных для отчета")
            return

        print("\n" + "=" * 60)
        print("Сводный отчет по всем сравнениям")
        print("=" * 60)

        # Таблица сравнения
        print("\nСравнение метрик по сценам:")
        print("-" * 120)
        print(f"{'Сцена':<20} {'MAE (мм)':<12} {'RMSE (мм)':<12} {'Abs Rel (%)':<12} "
              f"{'δ1 (%)':<12} {'Pearson':<12} {'Время стерео (с)':<15} {'Время DA (с)':<15}")
        print("-" * 120)

        for scene_name, results in self.comparison_results.items():
            metrics = results.get("comparison_metrics", {})
            stereo_time = results.get("stereo_time_seconds", 0)
            da_time = results.get("da_time_seconds", 0)

            print(f"{scene_name:<20} "
                  f"{metrics.get('mae', 0):<12.1f} "
                  f"{metrics.get('rmse', 0):<12.1f} "
                  f"{metrics.get('abs_rel', 0) * 100:<12.1f} "
                  f"{metrics.get('delta1', 0) * 100:<12.1f} "
                  f"{metrics.get('pearson_corr', 0):<12.3f} "
                  f"{stereo_time:<15.2f} "
                  f"{da_time:<15.2f}")

        print("-" * 120)

        # Средние значения
        if len(self.comparison_results) > 1:
            avg_mae = np.mean([r.get("comparison_metrics", {}).get("mae", 0)
                               for r in self.comparison_results.values()])
            avg_rmse = np.mean([r.get("comparison_metrics", {}).get("rmse", 0)
                                for r in self.comparison_results.values()])
            avg_abs_rel = np.mean([r.get("comparison_metrics", {}).get("abs_rel", 0)
                                   for r in self.comparison_results.values()]) * 100
            avg_delta1 = np.mean([r.get("comparison_metrics", {}).get("delta1", 0)
                                  for r in self.comparison_results.values()]) * 100
            avg_pearson = np.mean([r.get("comparison_metrics", {}).get("pearson_corr", 0)
                                   for r in self.comparison_results.values()])

            print(f"\nСредние значения:")
            print(f"  MAE: {avg_mae:.1f} мм")
            print(f"  RMSE: {avg_rmse:.1f} мм")
            print(f"  Abs Rel: {avg_abs_rel:.1f}%")
            print(f"  δ1: {avg_delta1:.1f}%")
            print(f"  Pearson корреляция: {avg_pearson:.3f}")

        # Выводы
        print("\n" + "=" * 60)
        print("Выводы:")
        print("=" * 60)

        print("""
        1. Depth Anything:
           - Преимущества:
             * Не требует калибровки
             * Работает с одиночными изображениями
             * Хорошо справляется со сложными текстурами
             * Быстрее на этапе инференса (после загрузки модели)

           - Недостатки:
             * Относительная (не метрическая) глубина
             * Требует масштабирования для метрических измерений
             * Может давать артефакты на границах объектов
             * Требует GPU для быстрой работы

        2. Стереопара:
           - Преимущества:
             * Метрическая точность (после калибровки)
             * Физически обоснованные измерения
             * Лучшая точность на однородных поверхностях
             * Не требует обучения/предобученных моделей

           - Недостатки:
             * Требует калибровки
             * Нужны два синхронизированных изображения
             * Плохо работает на однородных текстурах
             * Вычислительно сложные алгоритмы сопоставления

        3. Рекомендации:
           * Для метрически точных измерений: использовать стереопару
           * Для качественной оценки глубины на одиночных изображениях: Depth Anything
           * Для робототехники: комбинировать оба подхода
           * Для улучшения результатов: fine-tuning Depth Anything на своих данных
        """)

        # Сохранение отчета
        report_data = {
            "summary": {
                "total_scenes": len(self.comparison_results),
                "model_type": self.model_type,
                "device": str(self.device),
                "stereo_calibration": self.stereo_calibration_path,
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "scenes": self.comparison_results
        }

        with open("depth_comparison_report.json", "w") as f:
            json.dump(report_data, f, indent=4, default=str)

        print(f"\nПолный отчет сохранен в depth_comparison_report.json")


def main_depth_comparison():
    """Основная функция для Task 3"""
    print("=" * 60)
    print("Task 3: Сравнение Depth Anything со стереопарой")
    print("=" * 60)

    # Инициализация компаратора
    comparator = DepthAnythingComparator(
        stereo_calibration_path="stereo_calibration.json",
        model_type="small"  # Можно выбрать 'small', 'base', 'large'
    )

    # Проверка доступности Depth Anything
    if comparator.depth_anything_model is None:
        print("Depth Anything не доступна. Пропускаем Task 3...")
        return

    # Запуск сравнения
    try:
        comparator.run_comparison_for_backgrounds()

        # Дополнительное сравнение на других изображениях
        print("\n" + "=" * 60)
        print("Дополнительное сравнение на произвольных изображениях")
        print("=" * 60)

        # Пример с произвольными изображениями (если есть)
        test_images = glob.glob("test_images/*.jpg") + glob.glob("test_images/*.png")

        if test_images:
            for i, img_path in enumerate(test_images[:2]):  # Ограничимся 2 изображениями
                # Для произвольных изображений нужна стереопара
                # Здесь предполагаем, что у нас есть стереопара с тем же объектом
                stereo_pair = img_path.replace("test_images", "stereo")
                left_path = stereo_pair.replace(".jpg", "_left.jpg")
                right_path = stereo_pair.replace(".jpg", "_right.jpg")

                if os.path.exists(left_path) and os.path.exists(right_path):
                    scene_name = f"Тест_{i + 1}_{os.path.basename(img_path)}"
                    comparator.process_comparison(
                        stereo_left_path=left_path,
                        stereo_right_path=right_path,
                        da_image_path=img_path,
                        scene_name=scene_name
                    )

        print("\n" + "=" * 60)
        print("Task 3 завершена!")
        print("=" * 60)

    except Exception as e:
        print(f"Ошибка в Task 3: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_depth_comparison()