# YOLO Face Models

Поместите сюда сконвертированные модели:

- `yolov8n-face.tflite` - легкая модель (6MB), рекомендуется для мобильных
- `yolov8l-face.tflite` - большая модель (87MB), точнее но медленнее

## Как получить модели:

1. Скачайте исходные .pt модели:
```bash
# via GitHub CLI (если есть доступ)
gh release download 1.0.0 --repo akanametov/yolo-face --pattern "yolov8n-face.pt"
gh release download 1.0.0 --repo akanametov/yolo-face --pattern "yolov8l-face.pt"

# или напрямую
wget https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt
wget https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8l-face.pt
```

2. Конвертируйте в TFLite:
```bash
pip install ultralytics tensorflow
python convert_model.py --model yolov8n-face.pt --out app/src/main/assets/yolov8n-face.tflite
python convert_model.py --model yolov8l-face.pt --out app/src/main/assets/yolov8l-face.tflite
```

3. Или используйте готовые ONNX из HuggingFace:
- https://huggingface.co/deepghs/yolo-face
- https://huggingface.co/arnabdhar/YOLOv8-Face-Detection

## Автоматическая загрузка в CI:

GitHub Actions workflow автоматически скачивает и конвертирует модели при сборке.

Если моделей нет, приложение использует ML Kit Face Detection (работает из коробки).
