# AI Face Detector - Android

Приложение для Android с детектором лиц в реальном времени по камере.

## Возможности

- 📷 **Камера в реальном времени** - Preview с CameraX
- 🎯 **Детекция лиц** - Боксики + полигоны лица
- 🤖 **Два режима детекции:**
  - **ML Kit** (Google) - быстрый, работает из коробки, рисует контуры лица, глаза, нос, рот
  - **YOLOv8 Face** - использует модели YOLOv8n-face / YOLOv8l-face, конвертированные в TFLite
- 🔄 Переключение фронтальная/задняя камера
- 📊 FPS и счетчик лиц
- 🟩 Зеленые боксики вокруг лиц с процентами уверенности
- 🟣 Полигоны и лэндмарки лица (глаза, нос, рот)

## Модели YOLO

Поддерживаются модели из репозитория [akanametov/yolo-face](https://github.com/akanametov/yolo-face):

- `yolov8n-face.pt` - 6.2 MB, быстрая, для мобильных (рекомендуется)
- `yolov8l-face.pt` - 87.7 MB, точная, медленнее (ваша ссылка)
- `yolov8m-face.pt`, `yolov8s-face.pt` и другие

Ссылка из запроса:
```
https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8l-face.pt
```

## Структура проекта

```
ai-face/
├── app/
│   ├── src/main/
│   │   ├── java/com/aiface/detector/
│   │   │   ├── MainActivity.kt          # Основная активность с камерой
│   │   │   ├── OverlayView.kt           # Отрисовка боксиков и полигонов
│   │   │   ├── FaceDetectorHelper.kt    # ML Kit детектор
│   │   │   └── YoloFaceDetector.kt      # YOLOv8 TFLite детектор
│   │   ├── res/layout/activity_main.xml
│   │   └── assets/
│   │       ├── yolov8n-face.tflite      # (опционально) сконвертированная модель
│   │       └── yolov8l-face.tflite      # (опционально)
│   └── build.gradle.kts
├── convert_model.py                     # Скрипт конвертации .pt -> .tflite
├── .github/workflows/build-apk.yml      # Автосборка APK в GitHub Actions
└── README.md
```

## Быстрый старт

### 1. Скачать YOLO модель

```bash
# Через gh CLI
gh release download 1.0.0 --repo akanametov/yolo-face --pattern "yolov8n-face.pt"
gh release download 1.0.0 --repo akanametov/yolo-face --pattern "yolov8l-face.pt"

# Или wget
wget https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt
wget https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8l-face.pt
```

### 2. Конвертировать в TFLite

```bash
pip install ultralytics tensorflow

# Легкая модель (рекомендуется для Android)
python convert_model.py --model yolov8n-face.pt --out app/src/main/assets/yolov8n-face.tflite

# Большая модель (из вашего запроса)
python convert_model.py --model yolov8l-face.pt --out app/src/main/assets/yolov8l-face.tflite
```

### 3. Собрать APK

```bash
./gradlew assembleDebug
# APK будет в app/build/outputs/apk/debug/app-debug.apk
```

Или через Android Studio: Open Project -> Run

### 4. Автосборка через GitHub Actions

При пуше в `main` или `arena/*` автоматически:

1. Скачивает YOLO модели
2. Конвертирует в TFLite
3. Собирает Debug и Release APK
4. Загружает артефакты
5. Коммитит APK в репозиторий (`ai-face-detector.apk`)

Скачать APK: Actions -> Build APK -> Artifacts

## Как работает детекция

### ML Kit (по умолчанию)

- Использует `com.google.mlkit:face-detection`
- Находит лица, bounding box, 8 лэндмарок (глаза, уши, нос, рот)
- Контуры лица (овал, глаза, брови, губы, нос) - рисуются как полигоны
- Очень быстрый, работает на всех устройствах

### YOLOv8 Face

- Модель YOLOv8 обученная на датасете лиц
- Выход: `x, y, w, h, confidence, 5 keypoints (10 values)`
- Keypoints: левый глаз, правый глаз, нос, левый угол рта, правый угол рта
- Конвертирована в TFLite для работы на Android
- Поддерживает NMS и letterbox resize
- Точнее для маленьких лиц и в сложных условиях

## Технические детали

- **CameraX** 1.4.0 - для камеры
- **ML Kit** 16.1.7 - для детекции
- **TensorFlow Lite** 2.14.0 - для YOLO
- **ONNX Runtime** 1.17.0 - опционально для ONNX моделей
- **minSdk 24** (Android 7.0+), **targetSdk 34**
- **Kotlin** + **Coroutines**

## Установка APK

1. Скачайте `ai-face-detector.apk` из релизов или артефактов Actions
2. Включите "Установка из неизвестных источников" на Android
3. Установите APK
4. Дайте разрешение на камеру
5. Наведите камеру на лицо - увидите зеленый боксик и полигоны!

## Переключение режимов

- Кнопка **"Сменить камеру"** - фронт/зад
- Кнопка **"Режим"** - ML Kit <-> YOLOv8 (если модель загружена)

Если YOLO модель не найдена, используется ML Kit.

## Конвертация модели - детали

YOLOv8-face модели имеют выход:
- Бокс: `x_center, y_center, width, height` (нормализованы)
- Confidence: 0-1
- Keypoints: 5 точек x 2 координаты = 10 значений

Скрипт `convert_model.py` использует `ultralytics` для экспорта:

```python
from ultralytics import YOLO
model = YOLO("yolov8l-face.pt")
model.export(format='tflite', imgsz=640)
```

Модель 640x640, RGB, float32. На Android делается letterbox resize.

## Лицензия

MIT - используйте как хотите

## Автор

Сделано для задачи: Android Face Detector с YOLOv8l-face.pt
