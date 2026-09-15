"""
Convert YOLOv8 face .pt model to TFLite for Android
Supports yolov8n-face.pt, yolov8l-face.pt etc

Usage:
  python convert_model.py --model yolov8n-face.pt --out app/src/main/assets/yolov8n-face.tflite
  python convert_model.py --model yolov8l-face.pt --out app/src/main/assets/yolov8l-face.tflite

Requires: ultralytics, tensorflow
"""
import argparse
import os

def convert_to_tflite(pt_path, out_path, imgsz=640):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install ultralytics --quiet")
        from ultralytics import YOLO

    print(f"Loading model {pt_path}...")
    model = YOLO(pt_path)

    print(f"Exporting to TFLite with imgsz={imgsz}...")
    # Export to TFLite
    # For face detection, we need to keep NMS disabled and handle it manually, or export with NMS
    # We'll export as float16 for smaller size
    model.export(
        format='tflite',
        imgsz=imgsz,
        int8=False,
        half=False,  # Use float32 for better compatibility, or float16 for size
        dynamic=False,
        simplify=True
    )

    # The exported file will be in same dir as pt with .tflite extension
    base = os.path.splitext(pt_path)[0]
    exported = base + ".tflite"
    if os.path.exists(exported):
        import shutil
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.move(exported, out_path)
        print(f"Saved TFLite model to {out_path}")
        print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
    else:
        # Try to find in current dir
        for f in os.listdir("."):
            if f.endswith(".tflite"):
                print(f"Found {f}")
                import shutil
                shutil.move(f, out_path)
                break

def convert_to_onnx(pt_path, out_path, imgsz=640):
    try:
        from ultralytics import YOLO
    except ImportError:
        os.system("pip install ultralytics --quiet")
        from ultralytics import YOLO

    print(f"Loading model {pt_path}...")
    model = YOLO(pt_path)
    print(f"Exporting to ONNX...")
    model.export(format='onnx', imgsz=imgsz, dynamic=False, simplify=True)
    base = os.path.splitext(pt_path)[0]
    exported = base + ".onnx"
    if os.path.exists(exported):
        import shutil
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.move(exported, out_path)
        print(f"Saved ONNX model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n-face.pt", help="Path to .pt model")
    parser.add_argument("--out", type=str, default="app/src/main/assets/yolov8n-face.tflite", help="Output path")
    parser.add_argument("--format", type=str, default="tflite", choices=["tflite", "onnx"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    if args.format == "tflite":
        convert_to_tflite(args.model, args.out, args.imgsz)
    else:
        convert_to_onnx(args.model, args.out, args.imgsz)
