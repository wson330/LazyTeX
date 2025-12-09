"""Minimal predict entrypoint for the released TAMERICAL model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from model.tamerical_module import LitTAMERICAL
from datamodule import H_HI, W_HI, ScaleToLimitRange, vocab

# Paths kept together for quick editing when sharing the project.
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "image.png"
CKPT_PATH = BASE_DIR / "lightning_logs/version_10/checkpoints/epoch=117-step=130390.ckpt"
DICT_PATH = BASE_DIR / "lightning_logs/version_10/dictionary.txt"


def load_image(path: Path) -> Dict[str, torch.Tensor]:
    scaler = ScaleToLimitRange(w_lo=32, w_hi=W_HI, h_lo=32, h_hi=H_HI)
    image = Image.open(path).convert("L")
    np_img = scaler(np.array(image))
    np_img = np_img.astype("float32") / 255.0

    tensor = torch.zeros((1, H_HI, W_HI), dtype=torch.float32)
    mask = torch.ones((H_HI, W_HI), dtype=torch.bool)
    h, w = np_img.shape
    tensor[:, :h, :w] = torch.from_numpy(np_img)
    mask[:h, :w] = False
    return {"image": tensor.unsqueeze(0), "mask": mask.unsqueeze(0)}


def main() -> None:
    vocab.init(str(DICT_PATH))
    try:  # keep TAMER internals satisfied without requiring external setup
        from TAMER.tamer.datamodule import vocab as legacy_vocab

        legacy_vocab.word2idx = vocab.word2idx
        legacy_vocab.idx2word = vocab.idx2word
    except ModuleNotFoundError:
        pass
    model = LitTAMERICAL.load_from_checkpoint(str(CKPT_PATH), map_location="cpu")
    model.eval()

    batch = load_image(IMAGE_PATH)
    with torch.no_grad():
        hyps = model.approximate_joint_search(batch["image"], batch["mask"])
    best = hyps[0]
    tokens = [idx for idx in best.seq if idx not in (vocab.PAD_IDX, vocab.SOS_IDX, vocab.EOS_IDX)]
    latex = " ".join(vocab.indices2words(tokens))
    print(latex)


if __name__ == "__main__":
    main()
