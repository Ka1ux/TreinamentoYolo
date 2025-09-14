from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Treino com Ultralytics YOLO")
	p.add_argument("--data", default="data.yaml", help="caminho do data.yaml")
	p.add_argument("--model", default="yolo11n.pt", help="checkpoint inicial (ex: yolo11n.pt, yolov8s.pt)")
	p.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem")
	p.add_argument("--epochs", type=int, default=50)
	p.add_argument("--batch", type=int, default=16)
	p.add_argument("--name", default="exp", help="nome do experimento")
	p.add_argument("--device", default=0, help="GPU id ou 'cpu'")
	p.add_argument("--workers", type=int, default=8)
	p.add_argument("--project", default="runs/train", help="pasta de saida")
	return p.parse_args()


def main() -> int:
	try:
		from ultralytics import YOLO
	except Exception as e:
		print("[erro] Ultralytics não encontrado. Instale com: pip install -r requirements.txt", file=sys.stderr)
		raise

	args = parse_args()
	data_path = Path(args.data)
	if not data_path.exists():
		print(f"[erro] data.yaml não encontrado em {data_path.resolve()}", file=sys.stderr)
		return 2

	# Aviso rápido sobre estrutura do dataset
	try:
		import yaml
		cfg = yaml.safe_load(data_path.read_text())
		base = Path(cfg.get("path", data_path.parent))
		train_rel = cfg.get("train")
		val_rel = cfg.get("val") or cfg.get("valid")
		if train_rel:
			tr = (base / train_rel)
			if not tr.exists():
				print(f"[aviso] Caminho de treino não existe: {tr}. Ajuste 'path'/'train' no data.yaml.")
		if val_rel:
			vr = (base / val_rel)
			if not vr.exists():
				print(f"[aviso] Caminho de validação não existe: {vr}. Ajuste 'path'/'val' no data.yaml.")
	except Exception:
		pass

	model = YOLO(args.model)
	results = model.train(
		data=str(data_path),
		imgsz=args.imgsz,
		epochs=args.epochs,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		project=args.project,
		name=args.name,
		pretrained=True,
	)
	# Pequeno resumo
	try:
		best = results.best
		print(f"[ok] Treino finalizado. Pesos em: {best}")
	except Exception:
		pass
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
