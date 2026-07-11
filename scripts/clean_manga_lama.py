#!/usr/bin/env python3
"""Manga English-text removal via LaMa / manga model (lama-cleaner).

Tesseract detects English text (incl. vertical) -> mask -> inpaint.
Returns clean art with bubble outlines intact.

This is a legacy OCR-based engine. The AI-based `ctd_lama.py` is preferred
because it detects text more precisely (no screentone false positives, catches
small fonts and caption bars). Kept here as a fallback / alternative.

Run with:
    python3 clean_manga_lama.py <src> <dst> [--model lama|manga]
                                  [--dilate 10] [--jpn]

Notes / baked-in fixes (do NOT revert):
  * _map_ccw90 returns (y, W-1-x-bh, bh, bw)  -- correct vertical-text mapping.
  * min box height bh < 7 (was 12) -- captures 9px fonts in small bubbles.
  * min conf 50; psm 6 only for thin horizontal info bars (bh < vh*0.12, len>=3)
    so screentone grain is not read as text.
  * default dilate 10.
"""
import sys, argparse, json
import cv2
import numpy as np
import pytesseract
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler


def _map_cw90(x, y, bw, bh, H):
    # map a box (col x, row y, w, h) from a CW-90 rotated image back to original
    return (H - 1 - x - bw, y, bh, bw)


def _map_ccw90(x, y, bw, bh, W):
    # rotate CCW90 maps (x,y) in rotated img -> (y, W-1-x-bh) in original
    return (y, W - 1 - x - bh, bh, bw)


def build_mask(img, dilate=14, jpn=False):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rot_cw = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_ccw = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    variants = [(img, None, 0, 0), (rot_cw, "cw", h, w), (rot_ccw, "ccw", h, w)]
    lang = "eng+jpn" if jpn else "eng"
    for vimg, kind, Hp, Wp in variants:
        vh, vw = vimg.shape[:2]
        # sparse psm (11/12) for normal text. psm 6 (block) catches the
        # small-caps info bars that 11/12 miss; guarded by tight size filter
        # so screentone grain is not read as text.
        for psm in (11, 12, 6):
            data = pytesseract.image_to_data(
                vimg, output_type=pytesseract.Output.DICT,
                config=f"--psm {psm} --oem 1 -l {lang}")
            for i in range(len(data["text"])):
                t = (data["text"][i] or "").strip()
                try:
                    c = int(data["conf"][i])
                except Exception:
                    c = -1
                if c < 50 or len(t) < 2:
                    continue
                x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                # tighter filters: drop tiny/noisy boxes and huge blocks.
                # psm 6 tends to read screentone as text -> only accept it for
                # long, thin horizontal bars (info/translation bars), with high
                # confidence and a real word, so screentone grain is rejected.
                if psm == 6:
                    if not (bh < vh * 0.12 and c >= 50 and len(t) >= 3):
                        continue
                else:
                    if bw < 8 or bh < 7 or bw > vw * 0.5 or bh > vh * 0.4:
                        continue
                if kind == "cw":
                    nx, ny, nbw, nbh = _map_cw90(x, y, bw, bh, Hp)
                elif kind == "ccw":
                    nx, ny, nbw, nbh = _map_ccw90(x, y, bw, bh, Wp)
                else:
                    nx, ny, nbw, nbh = x, y, bw, bh
                cv2.rectangle(mask,
                              (max(0, nx - 3), max(0, ny - 3)),
                              (min(w, nx + nbw + 3), min(h, ny + nbh + 3)),
                              255, -1)
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
    return mask


def run(src, dst, model_name="lama", dilate=14, jpn=False):
    img = cv2.imread(src)
    if img is None:
        print(json.dumps({"ok": False, "error": f"cannot read {src}"}))
        sys.exit(1)
    mask = build_mask(img, dilate=dilate, jpn=jpn)
    px = int((mask > 0).sum())
    total = img.shape[0] * img.shape[1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = ModelManager(name=model_name, device="cpu")
    config = Config(
        ldm_steps=25,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.CROP,
        hd_strategy_resize_limit=1024,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_crop_margin=64,
        zits_wireframe=False,
    )
    result = model(rgb, mask, config)
    out = result if isinstance(result, np.ndarray) else cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst, out)
    print(json.dumps({
        "ok": True,
        "model": model_name,
        "mask_pct": round(100 * px / total, 1),
        "mask_px": px,
        "dst": dst,
    }))
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", default="lama", choices=["lama", "manga"])
    ap.add_argument("--dilate", type=int, default=14)
    ap.add_argument("--jpn", action="store_true", help="also OCR Japanese (catches SFX kana)")
    args = ap.parse_args()
    run(args.src, args.dst, args.model, dilate=args.dilate, jpn=args.jpn)
