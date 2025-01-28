import time
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage import measure

# Для замера времени выполнения
start_time = time.time()

print("📥 Начало загрузки GeoTIFF...")
file_path = r"C:\Users\Acer\Desktop\EO2\test1.tif"

with rasterio.open(file_path) as src:
    image_data = src.read()
    profile = src.profile
    transform = src.transform
    height, width = src.height, src.width

print(f"✅ Файл загружен. Время: {time.time() - start_time:.2f} сек.")

# 📌 Извлечение данных
VV = image_data[0].astype(np.float32).flatten()
VH = image_data[2].astype(np.float32).flatten()
coh_VV = image_data[1].astype(np.float32).flatten()
coh_VH = image_data[3].astype(np.float32).flatten()

# 📌 Вычисляем NDFI
print("🧮 Вычисляем NDFI...")
ndfi = (VV - VH) / (VV + VH + 1e-6)  # Добавляем малую константу, чтобы избежать деления на 0

# 📌 Создаем матрицу признаков
features = np.column_stack((VV, VH, coh_VV, coh_VH, ndfi))

# 📌 Исключаем пиксели с NoData (VV, VH, Coherence > 0)
mask = (VV > 0) & (VH > 0) & (coh_VV > 0) & (coh_VH > 0)
features = features[mask]
ndfi = ndfi[mask]

# 📌 Определяем классы затопления по VV
labels = np.zeros_like(VV, dtype=np.uint8)
threshold_moderate = np.percentile(VV[mask], 70)
threshold_high = np.percentile(VV[mask], 90)

labels[VV > threshold_moderate] = 1  # Умеренное затопление
labels[VV > threshold_high] = 2  # Сильное затопление
labels = labels[mask]

print(f"📊 Классы затопления распределены. Время: {time.time() - start_time:.2f} сек.")

# 📌 Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 🔹 Ограничиваем количество пикселей для тренировки (снижаем нагрузку)
num_samples = min(500000, len(X_train))  # Берем максимум 500K пикселей
indices = np.random.choice(len(X_train), num_samples, replace=False)

X_train_subset = X_train[indices]
y_train_subset = y_train[indices]

print(f"📉 Обучаем модель на {num_samples} пикселях.")

# 🔹 Обучаем RandomForest с ограничением ресурсов
clf = RandomForestClassifier(n_estimators=30, max_depth=8, n_jobs=-1, random_state=42)

# 🔹 Обучение модели батчами
batch_size = 100000
for i in range(0, len(X_train_subset), batch_size):
    X_batch = X_train_subset[i:i + batch_size]
    y_batch = y_train_subset[i:i + batch_size]

    print(f"🔄 Обучение батча {i // batch_size + 1}/{len(X_train_subset) // batch_size + 1}")
    clf.fit(X_batch, y_batch)

print(f"✅ Обучение завершено. Время: {time.time() - start_time:.2f} сек.")

# 📌 Оценка модели
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Точность модели: {accuracy:.4f}")

# 📌 Сохранение классифицированного изображения
print("🖼️ Сохраняем классифицированное изображение...")
predicted_labels = clf.predict(features)
classified_image = np.zeros(VV.shape, dtype=np.uint8)
classified_image[mask] = predicted_labels

output_file = "flood_classification_ndfi.tif"
with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(classified_image.reshape(height, width), 1)

print(f"✅ Классифицированное изображение сохранено в {output_file}.")

# 📌 Создание маски затопленных областей
flood_mask = np.zeros_like(classified_image, dtype=np.uint8)
flood_mask[classified_image == 1] = 1  # Умеренное затопление
flood_mask[classified_image == 2] = 2  # Сильное затопление

# 📌 Визуализация маски затопления
plt.figure(figsize=(10, 6))
plt.imshow(flood_mask.reshape(height, width), cmap="Reds", interpolation="nearest")
plt.colorbar(label="0 = Суша, 1 = Умеренное, 2 = Сильное")
plt.title("Маска затопленных зон")
plt.show()

# 📌 Обнаружение контуров затопления
if np.any(flood_mask):  # Проверяем, есть ли затопленные пиксели
    contours = measure.find_contours(flood_mask.reshape(height, width), 0.5)

    # 📌 Визуализация контуров затопления
    plt.figure(figsize=(10, 6))
    plt.imshow(classified_image.reshape(height, width), cmap='gray', interpolation='nearest')
    plt.colorbar(label="Классификация (0=суша, 1=умеренно, 2=сильно)")
    plt.title("Контуры затопленных зон")

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    plt.show()
else:
    print("⚠️ Нет затопленных зон, контуры не найдены.")

# 📌 Сохранение маски затопления в TIFF
output_flood_mask = "flood_mask.tif"
profile.update(dtype=rasterio.uint8, count=1)

with rasterio.open(output_flood_mask, 'w', **profile) as dst:
    dst.write(flood_mask.reshape(height, width), 1)

print(f"✅ Файл с маской затопления сохранен в {output_flood_mask}")

