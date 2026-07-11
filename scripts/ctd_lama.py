#!/usr/bin/env python3
"""Engine: use PanelCleaner's comic-text-detector (CTD) to generate a precise
text mask (covers bubbles, text boxes, and vertical text), then inpaint with
LaMa (lama_cleaner, HDStrategy.CROP for quality). Removes ALL text.

This is the recommended engine: AI text detection (YOLO) is far more accurate
than Tesseract OCR, and it captures narration boxes / caption bars / vertical
text that PanelCleaner's own pipeline skips (PanelCleaner intersects the text
mask with the bubble-outline mask, so narration boxes survive).

Run with:
    python3 ctd_lama.py <src> <dst> [--device cpu] [--dilate 8]

Environment (this machine's layout is the default, override if needed):
    PCLEANER_VENV  path to the venv that has `pcleaner`  (default /root/pcleaner-venv)
    IMGTOOLS_VENV  path to the venv that has `lama_cleaner` (default /root/imgtools)
    CTD_MODEL      path to comictextdetector.pt.onnx
                   (default $HOME/.cache/pcleaner/model/comictextdetector.pt.onnx)
"""
import sys, os, json, argparse, tempfile, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# --- venv wiring (override via env vars; defaults match this host) -----------
PCLEANER_VENV = os.environ.get("PCLEANER_VENV", "/root/pcleaner-venv")
IMGTOOLS_VENV = os.environ.get("IMGTOOLS_VENV", "/root/imgtools")
_pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _v in (PCLEANER_VENV, IMGTOOLS_VENV):
    _sp = os.path.join(_v, "lib", _pyver, "site-packages")
    if os.path.isdir(_sp):
        sys.path.insert(0, _sp)

import pcleaner.config as cfg
from pcleaner.ctd_interface import model2annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dilate", type=int, default=8)
    args = ap.parse_args()

    src = args.src
    if not os.path.exists(src):
        print(json.dumps({"ok": False, "error": f"no src {src}"}))
        sys.exit(1)

    workdir = Path(tempfile.mkdtemp(prefix="ctd_"))
    base = os.path.basename(src)
    shutil.copy(src, workdir / base)

    gen = cfg.GeneralConfig(input_height_lower_target=0,
                            input_height_upper_target=99999,
                            split_long_strips=False)
    det = cfg.TextDetectorConfig(concurrent_models=1)
    model_path = Path(os.environ.get(
        "CTD_MODEL",
        os.path.expanduser("~/.cache/pcleaner/model/comictextdetector.pt.onnx")))
    out = workdir / "ctd"
    out.mkdir(exist_ok=True)
    model2annotations(gen, det, model_path, [workdir / base], out)

    mask_path = None
    for f in out.iterdir():
        if f.name.endswith("_raw_mask.png"):
            mask_path = f
            break
    if mask_path is None:
        print(json.dumps({"ok": False, "error": "no ctd mask"}))
        sys.exit(1)

    mask = np.array(Image.open(mask_path).convert("L"))
    orig = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
    # CTD may internally resize; force mask back to original dims.
    if mask.shape[:2] != orig.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(
            (orig.shape[1], orig.shape[0]), Image.NEAREST))

    mask = cv2.dilate(mask, np.ones((args.dilate, args.dilate), np.uint8),
                      iterations=1)
    mask_bin = (mask > 128).astype(np.uint8) * 255

    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config, HDStrategy, LDMSampler
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    model = ModelManager(name="lama", device=args.device)
    config = Config(ldm_steps=25, ldm_sampler=LDMSampler.ddim,
                    hd_strategy=HDStrategy.CROP, hd_strategy_resize_limit=1024,
                    hd_strategy_crop_trigger_size=512, hd_strategy_crop_margin=64,
                    zits_wireframe=False)
    result = model(rgb, mask_bin, config)
    out_img = (result if isinstance(result, np.ndarray)
               else cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))
    # ensure exact original size
    if out_img.shape[:2] != orig.shape[:2]:
        out_img = cv2.resize(out_img, (orig.shape[1], orig.shape[0]))
    cv2.imencode('.png', out_img)[1].tofile(args.dst)

    px = int((mask_bin > 0).sum())
    total = orig.shape[0] * orig.shape[1]
    print(json.dumps({"ok": True, "engine": "ctd-lama",
                      "mask_pct": round(100 * px / total, 1), "dst": args.dst}))


if __name__ == "__main__":
    main()
