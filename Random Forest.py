import time
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Для визуализации матрицы ошибок
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from skimage import measure

# Замер времени
start_time = time.time()

# 1. Загрузка GeoTIFF-файла
print("📥 Начало загрузки GeoTIFF...")
file_path = r"C:\Users\Acer\Desktop\EO2\test1.tif"

with rasterio.open(file_path) as src:
    image_data = src.read()      # Читаем все каналы
    profile = src.profile        # Сохраняем профиль для последующей записи
    transform = src.transform    # Для расчёта размера пикселя
    height, width = src.height, src.width

print(f"✅ Файл загружен. Время: {time.time() - start_time:.2f} сек.")

# 2. Извлечение данных из каналов
# Предполагается, что каналы расположены так: [VV, coh_VV, VH, coh_VH]
VV      = image_data[0].astype(np.float32).flatten()
coh_VV  = image_data[1].astype(np.float32).flatten()
VH      = image_data[2].astype(np.float32).flatten()
coh_VH  = image_data[3].astype(np.float32).flatten()

# 3. Вычисляем NDFI (Normalized Difference Flood Index)
print("🧮 Вычисляем NDFI...")
ndfi = (VV - VH) / (VV + VH + 1e-6)  # добавляем константу, чтобы избежать деления на 0

# 4. Формирование матрицы признаков
# Используем признаки: VV, VH, coh_VV, coh_VH, ndfi
features = np.column_stack((VV, VH, coh_VV, coh_VH, ndfi))

# Исключаем пиксели с отсутствующими данными (значения <= 0)
mask = (VV > 0) & (VH > 0) & (coh_VV > 0) & (coh_VH > 0)
features = features[mask]

# 5. Определение классов
# Обычно водная поверхность имеет низкие значения VV.
# Здесь устанавливаем пороги (на основе 30-го и 10-го процентиля):
# 0 – Суша, 1 – Умеренное затопление, 2 – Сильное затопление
threshold_moderate = np.percentile(VV[mask], 30)  # порог для умеренного затопления
threshold_strong   = np.percentile(VV[mask], 10)   # порог для сильного затопления

labels = np.zeros_like(VV, dtype=np.uint8)  # по умолчанию 0 = Суша
labels[VV < threshold_moderate] = 1         # если значение ниже порога – Умеренное затопление
labels[VV < threshold_strong]   = 2         # если значение ещё ниже – Сильное затопление
labels = labels[mask]

print(f"📊 Классы распределены. Время: {time.time() - start_time:.2f} сек.")
print(f"Порог умеренного затопления (30-й процентиль): {threshold_moderate:.4f}")
print(f"Порог сильного затопления (10-й процентиль): {threshold_strong:.4f}")

# 6. Разделение данных: 70% для обучения, 30% для тестирования
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Ограничиваем число пикселей для обучения (до 500000), чтобы снизить нагрузку
num_samples = min(500000, len(X_train))
indices = np.random.choice(len(X_train), num_samples, replace=False)
X_train_subset = X_train[indices]
y_train_subset = y_train[indices]

print(f"📉 Обучаем модель на {num_samples} пикселях.")

# 7. Обучение модели RandomForest
clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)
clf.fit(X_train_subset, y_train_subset)

print(f"✅ Обучение завершено. Время: {time.time() - start_time:.2f} сек.")

# 8. Accuracy Assessment на тестовой выборке
y_pred = clf.predict(X_test)
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Общая точность (Overall Accuracy): {overall_accuracy*100:.2f}%")

# Коэффициент Каппа
kappa = cohen_kappa_score(y_test, y_pred)
print(f"Коэффициент Каппа (Kappa Coefficient): {kappa:.4f}")

# Матрица ошибок (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Отчет по классификации
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Вычисляем точность производителя (Producer's Accuracy, PA) и пользователя (User's Accuracy, UA)
pa = np.diag(cm) / np.sum(cm, axis=1)
ua = np.diag(cm) / np.sum(cm, axis=0)
for i, (p, u) in enumerate(zip(pa, ua)):
    print(f"Class {i}: Producer's Accuracy = {p*100:.2f}%, User's Accuracy = {u*100:.2f}%")

# Графическая визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
class_names = ["Суша", "Умеренное затопление", "Сильное затопление"]
# Используем colormap "YlOrRd", чтобы цвета были более выразительными
sns.heatmap(cm, annot=True, fmt='d', cmap="YlOrRd", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Предсказанные метки")
plt.ylabel("Истинные метки")
plt.title(f"Confusion Matrix\nOverall Accuracy = {overall_accuracy*100:.2f}%")
plt.show()

# 9. Классификация всего изображения батчами (для экономии памяти)
print("\n🖼️ Классифицируем изображение по частям...")
chunk_size = 1_000_000  # Размер батча; корректируйте при необходимости
num_chunks = (features.shape[0] + chunk_size - 1) // chunk_size
predicted_labels = np.empty(features.shape[0], dtype=np.uint8)

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min(start_idx + chunk_size, features.shape[0])
    print(f"Обработка батча {i+1} из {num_chunks}...")
    predicted_labels[start_idx:end_idx] = clf.predict(features[start_idx:end_idx])

# Восстанавливаем классифицированное изображение в исходной форме
classified_image = np.zeros(VV.shape, dtype=np.uint8)
classified_image[mask] = predicted_labels

# 10. Сохранение классифицированного изображения в TIFF
output_file = "flood_classification_ndfi.tif"
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(classified_image.reshape(height, width), 1)

print(f"\n✅ Классифицированное изображение сохранено в {output_file}.")

# 11. Создание маски затопления для визуализации
# Здесь: 0 = Суша, 1 = Умеренное затопление, 2 = Сильное затопление
flood_mask = np.copy(classified_image)

# 12. Расчёт площадей по классам
# Определяем размер пикселя (пиксельная ширина и высота)
pixel_width  = abs(transform.a)
pixel_height = abs(transform.e)  # Обычно transform.e отрицательный, поэтому берем abs
area_per_pixel = pixel_width * pixel_height  # в кв. метрах

# Рассчитываем число пикселей для каждого класса (учитывая только валидные пиксели, где mask=True)
non_flooded_pixels    = np.count_nonzero(classified_image[mask] == 0)
moderate_flooded_pixels = np.count_nonzero(classified_image[mask] == 1)
strong_flooded_pixels   = np.count_nonzero(classified_image[mask] == 2)
total_pixels = non_flooded_pixels + moderate_flooded_pixels + strong_flooded_pixels

non_flooded_area    = non_flooded_pixels * area_per_pixel
moderate_flooded_area = moderate_flooded_pixels * area_per_pixel
strong_flooded_area   = strong_flooded_pixels * area_per_pixel
total_area = non_flooded_area + moderate_flooded_area + strong_flooded_area

non_flooded_perc    = (non_flooded_area / total_area) * 100
moderate_flooded_perc = (moderate_flooded_area / total_area) * 100
strong_flooded_perc   = (strong_flooded_area / total_area) * 100

print("\n🗺️ Площадь по классам (только для валидных пикселей):")
print(f"Общая площадь: {total_area:,.2f} кв.м.")
print(f"Суша: {non_flooded_area:,.2f} кв.м. ({non_flooded_perc:.2f}%)")
print(f"Умеренное затопление: {moderate_flooded_area:,.2f} кв.м. ({moderate_flooded_perc:.2f}%)")
print(f"Сильное затопление: {strong_flooded_area:,.2f} кв.м. ({strong_flooded_perc:.2f}%)")

# 13. Визуализация маски затопления
plt.figure(figsize=(10, 6))
plt.imshow(flood_mask.reshape(height, width), cmap="YlGnBu", interpolation="nearest")
plt.colorbar(label="0 = Суша, 1 = Умеренное затопление, 2 = Сильное затопление")
plt.title("Затопленные области")
plt.show()

# 14. Обнаружение и визуализация контуров затопленных зон
if np.any(flood_mask):
    contours = measure.find_contours(flood_mask.reshape(height, width), 0.5)
    plt.figure(figsize=(10, 6))
    plt.imshow(classified_image.reshape(height, width), cmap='gray', interpolation='nearest')
    plt.colorbar(label="Классификация (0 = Суша, 1 = Умеренное затопление, 2 = Сильное затопление)")
    plt.title("Контуры затопленных зон")
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')
    plt.show()
else:
    print("⚠️ Нет затопленных зон, контуры не найдены.")

# 15. Сохранение маски затопления в отдельный TIFF-файл
output_flood_mask = "flood_mask.tif"
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open(output_flood_mask, 'w', **profile) as dst:
    dst.write(flood_mask.reshape(height, width), 1)

print(f"✅ Файл с маской затопления сохранен в {output_flood_mask}")
