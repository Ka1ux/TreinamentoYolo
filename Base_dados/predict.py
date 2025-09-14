from __future__ import annotations
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Inferência com Ultralytics YOLO")
    p.add_argument("--weights", default="auto", help="caminho para pesos .pt ou 'auto' para detectar o mais recente")
    p.add_argument("--source", default="data/images/valid/images", help="imagem/pasta/video ou glob")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", action="store_true", help="salvar resultados anotados")
    return p.parse_args()


def main() -> int:
    from ultralytics import YOLO
    args = parse_args()
    # Resolver pesos: se 'auto' ou caminho inexistente, procurar o mais recente em runs/train/*/weights/best.pt
    weights_path = Path(args.weights)
    if args.weights == "auto" or not weights_path.exists():
        candidates = sorted(Path("runs/train").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            weights_path = candidates[0]
            print(f"[info] Usando pesos: {weights_path}")
        else:
            raise FileNotFoundError("Nenhum best.pt encontrado em runs/train/*/weights/. Treine o modelo primeiro (train.py).")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
    )
    # imprimir contagem por imagem
    try:
        for r in results:
            p: Path = Path(getattr(r, "path", ""))
            print(f"{p.name}: {len(r.boxes) if hasattr(r, 'boxes') else 0} detecções")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
