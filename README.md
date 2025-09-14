# YOLO Training – Ultralytics YOLOv8 🚀

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/) 
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)](https://docs.ultralytics.com/)

Train and run object detection models using the **Ultralytics YOLOv8** API with your custom dataset.

---

## ⚙️ Installation

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```
## 🧪 Training
```
Quick test (1 epoch):
```
python3 train.py --data data.yaml --model yolov8n.pt --epochs 1 --batch 8 --imgsz 640 --name test --device cpu

```
Full training example:
```
python3 train.py --data data.yaml --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --name exp --device 0
```

Trained weights: runs/train/<name>/weights/best.pt
```
## 🔍 Inference

Run predictions:
```
python3 predict.py --source Base_dados/images/valid --save --device cpu
```

Specify custom weights:
```
python3 predict.py --weights runs/train/exp/weights/best.pt --source Base_dados/images/valid --save --device cpu


Predictions are saved in runs/detect/.
```
## 📦 Export Model

Export trained model:
```
python3 export.py --weights runs/train/exp/weights/best.pt --format onnx --device cpu


Supported formats: onnx, engine (TensorRT), ncnn, torchscript, openvino, coreml.
```
## 📝 Dataset Notes

For each image.jpg, create image.txt with YOLO labels:
```
class x_center y_center width height


(all normalized [0,1])

Update nc and names in data.yaml for your classes.
```
## 📌 References

[Ultralytics YOLO Documentation](https://docs.ultralytics.com/)

[YOLO Official Repository](https://github.com/ultralytics/)
