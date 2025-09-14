from __future__ import annotations
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
DATA = ROOT / "data"


def sync_split(split: str) -> None:
    # algumas bases usam 'val' em vez de 'valid'
    split_alt = "val" if split == "valid" else split

    # origem: estrutura atual do usuário: data/images/<split>/labels/*.txt
    src_labels = DATA / "images" / split / "labels"
    if not src_labels.exists():
        # tentar alternativa data/images/val/labels quando split='valid'
        src_labels = DATA / "images" / split_alt / "labels"

    # destino: formato Ultralytics: data/labels/<split>/*.txt
    dst_labels = DATA / "labels" / (split if (DATA / "images" / split).exists() else split_alt)
    dst_labels.mkdir(parents=True, exist_ok=True)
    if not src_labels.exists():
        print(f"[aviso] Não encontrei rótulos em {src_labels}")
        return
    count = 0
    for txt in src_labels.rglob("*.txt"):
        # mantém apenas o nome do arquivo (sem subpastas extras)
        target = dst_labels / txt.name
        shutil.copy2(txt, target)
        count += 1
    print(f"[{split}] copiados {count} arquivos .txt para {dst_labels}")


def main() -> int:
    # cria estrutura esperada de pastas
    (DATA / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATA / "labels" / "valid").mkdir(parents=True, exist_ok=True)
    # sincroniza labels de acordo com layout atual do usuário
    for split in ("train", "valid"):
        sync_split(split)
    print("[ok] Dataset preparado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
