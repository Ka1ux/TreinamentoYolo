from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Exportar modelo YOLO (Ultralytics)")
    p.add_argument("--weights", default="auto", help="caminho para pesos .pt ou 'auto' para usar o mais recente")
    p.add_argument("--format", default="onnx", choices=[
        "onnx", "engine", "ncnn", "torchscript", "openvino", "coreml"
    ])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> int:
    from ultralytics import YOLO
    a = parse_args()

    weights_path = Path(a.weights)
    if a.weights == "auto" or not weights_path.exists():
        candidates = sorted(Path("runs/train").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError("Nenhum best.pt encontrado em runs/train/*/weights. Treine o modelo antes de exportar.")
        weights_path = candidates[0]
        print(f"[info] Usando pesos: {weights_path}")

    model = YOLO(str(weights_path))
    out = model.export(format=a.format, imgsz=a.imgsz, device=a.device)
    print(f"Exportado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
