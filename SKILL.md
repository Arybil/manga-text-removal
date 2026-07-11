---
name: manga-text-removal
description: "Strip English/Japanese text (speech bubbles, narration boxes, caption bars, vertical text) from manga panels for re-lettering. Three engines: CTD-LaMa (best, AI detection via PanelCleaner comic-text-detector + LaMa inpaint), LaMa OCR (Tesseract legacy), PanelCleaner (bubbles only). Provides CLI scripts, venv setup, FastAPI web integration. Use when user wants to remove/clean/erase text from manga images, build a manga text-cleaning web tool, or integrate PanelCleaner."
---

# Manga Text Removal Pipeline

Removes ALL text from manga pages (bubbles, narration boxes, info/caption bars,
vertical text) leaving clean art for re-lettering. Three engines, all inpaint via
LaMa (lama_cleaner, HDStrategy.CROP).

## Engine comparison (verified on real manga page, 1024x722)

| Engine | Default | Text left (OCR) | White artifact | Speed |
|---|---|---|---|---|
| **ctd-lama** | yes | 0 (bubbles+boxes+caption+vertical) | 2.57% | ~20s |
| lama (OCR/Tesseract) | no | 0 (after fixes) | 3.22% | ~15s |
| panelcleaner | no | MANY (boxes/vertical escape) | ~0% | ~8s |

CTD-LaMa wins: AI text detection (YOLO comic-text-detector from PanelCleaner repo)
is far more precise than Tesseract heuristics; covers text in boxes/captions that
PanelCleaner deliberately skips (PanelCleaner intersects text mask with bubble-outline
mask, so narration boxes without a bubble outline survive).

## Setup (one-time)

Two venvs:
- `imgtools` venv — has `lama_cleaner`, `torch`, `cv2`, `pytesseract`, `Pillow`.
- `pcleaner-venv` venv — `pcleaner==2.11.11` (brings `comic-text-detector`,
  `manga_ocr`, `simple_lama_inpainting`, `transformers`).

`ctd_lama.py` runs under the `imgtools` venv but inserts pcleaner's site-packages
into sys.path to import `pcleaner.config` + `pcleaner.ctd_interface`. Works when
both are the same python minor version and the CTD model is cached at
`~/.cache/pcleaner/model/comictextdetector.pt.onnx`.

Install pcleaner:
```bash
python3 -m venv pcleaner-venv
pcleaner-venv/bin/pip install -U pip
pcleaner-venv/bin/pip install "pcleaner==2.11.11"
# transformers 5.x breaks manga_ocr import; downgrade if needed:
pcleaner-venv/bin/pip install "transformers<5"
# first run downloads comictextdetector.pt.onnx (~tens of MB) into ~/.cache/pcleaner/model/
```

imgtools venv must have lama_cleaner. If missing:
```bash
imgtools/bin/pip install lama-cleaner opencv-python-headless pytesseract pillow numpy
```

FastAPI web needs `python-multipart` (for UploadFile Form):
```bash
pip install python-multipart
```

## Scripts (in scripts/)

### ctd_lama.py (PRIMARY — CTD-LaMa)
Run: `python3 scripts/ctd_lama.py <src> <dst> [--device cpu] [--dilate 8]`
- Uses `model2annotations` from pcleaner to produce `<name>_raw_mask.png` (AI text
  mask covering ALL text, not just bubble interiors).
- Resizes mask to original size, dilates (default 8), inpaints with lama_cleaner
  HDStrategy.CROP, resizes result back to exact original size.
- Prints JSON: `{"ok":true,"engine":"ctd-lama","mask_pct":7.6,"dst":"..."}`
- Env overrides: `PCLEANER_VENV`, `IMGTOOLS_VENV`, `CTD_MODEL`.
- Key gotcha: CTD may internally resize the image; always resize mask AND final
  result back to original (cv2) dims, else size mismatch.

### clean_manga_lama.py (legacy Tesseract OCR engine)
Run: `python3 scripts/clean_manga_lama.py <src> <dst> [--model lama|manga] [--dilate 10] [--jpn]`
Critical fixes baked in (do NOT revert):
- Rotation mapping `_map_ccw90` returns `(y, W-1-x-bh, bh, bw)` (NOT `H-1-y-bh`).
- Min box height `bh < 7` (was 12) — captures 9px fonts in small bubbles.
- Min conf `50`; psm 6 only for thin horizontal info bars (`bh < vh*0.12`, `len>=3`)
  to avoid reading screentone as text.
- Default dilate 10.
- JSON output (needs `import json` at top).

### pclean_cli.py (PanelCleaner bubbles-only engine)
Run: `python3 scripts/pclean_cli.py <src> <dst> [--device cpu]`
- Calls `pcleaner clean`, finds `<name>_clean.<ext>` in `--output_dir`,
  copies to dst, prints JSON. Mask only covers speech bubbles (~1.6%).
- Env override: `PCLEANER_BIN`.
- Use as an option when user only wants bubble text removed.

## FastAPI integration (web panel)

Endpoint (must use `Form(...)` for all params — curl `-F` won't pass query params):
```python
@app.post("/api/clean")
async def clean_page(
    file: UploadFile = File(...),
    model: str = Form("lama"),
    dilate: int = Form(10),
    jpn: bool = Form(False),
    engine: str = Form("ctd-lama")):
    import asyncio, json
    # save upload to a temp .png
    ...
    if engine == "panelcleaner":
        script, params = PCLEANER_SCRIPT, [src, dst, "--device","cpu"]
    elif engine == "ctd-lama":
        script, params = CTD_SCRIPT, [src, dst, "--device","cpu","--dilate",str(dilate)]
    else:  # lama OCR
        script, params = LAMA_SCRIPT, [src, dst, "--model",model,
                                       "--dilate",str(dilate)] + (["--jpn"] if jpn else [])
    proc = await asyncio.create_subprocess_exec(
        "/path/to/imgtools/bin/python", script, *params,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    res = json.loads(out.decode().splitlines()[-1])  # last line is the JSON
    return {"ok": res["ok"], "mask_pct": res.get("mask_pct"),
            "file": os.path.basename(res["dst"]) if res["ok"] else None,
            "engine": engine, "model": model if engine!="panelcleaner" else "panelcleaner"}
```
Serve cleaned file via `/api/clean/file?name=...`. Always use
`asyncio.create_subprocess_exec` + read full stdout (not `to_thread(subprocess.run)`)
for robust JSON capture on Python 3.11.

Frontend: engine `<select>` with options `ctd-lama` (default), `panelcleaner`, `lama`;
for lama show model select + jpn checkbox + dilate slider; for ctd-lama show only
dilate slider. After upload, POST FormData to `/api/clean`, show before/after `<img>`,
download link `/api/clean/file?name=<file>`.

## Verification method (IMPORTANT)

Do NOT trust vision_analyze / VLM for "is text gone?" — it hallucinates text in
clean bubbles. Use OCR as ground truth:
```bash
python3 - <<'PY'
import cv2, pytesseract, numpy as np
def ocr(p):
    img = cv2.imread(p)
    txt=[]
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        r = cv2.rotate(img,rot) if rot else img
        for psm in (11,12,6):
            d=pytesseract.image_to_data(r,output_type=pytesseract.Output.DICT,config=f"--psm {psm} -l eng")
            for i,t in enumerate(d["text"]):
                t=(t or "").strip()
                try: c=int(d["conf"][i])
                except: c=-1
                if c>=50 and len(t)>=2: txt.append(t.upper())
    return txt
txt_o=ocr("clean.png")
left=[w for w in txt_o if w not in ("A","I","THE")]
print("LEFTOVER LEGIT:", left)
clean=cv2.imread("clean.png"); gray=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
print("WHITE ARTIFACT: %.2f%%" % (100*(gray>245).mean()))
PY
```
Target: `LEFTOVER LEGIT: []`, white artifact < 4%.

## Pitfalls (env-specific)

- `ps -eC chrome` is BROKEN on some hosts (false-positives matching kernel/systemd
  PIDs). Use `pgrep -fc chrome` or scan `/proc/*/cmdline` to count real browsers.
- Gateway OOM is caused by the Python gateway itself hitting ~4.5GB RSS while
  processing images (vision_analyze + text removal), NOT by chrome. Set a
  `MemoryMax` on the gateway service to prevent future OOM kills.
- `pcleaner` outputs `<name>_clean.<ext>` directly in `--output_dir` (no `cleaned/`
  subfolder) in newer versions; search both `workdir` and `workdir/cleaned`.
- `model2annotations(gen, det, model_path, [img_paths], out_dir)` — create
  `GeneralConfig(input_height_lower_target=0, input_height_upper_target=99999,
  split_long_strips=False)` and `TextDetectorConfig(concurrent_models=1)`.
- comic-text-detector raw mask = `<name>_raw_mask.png` in out_dir.
- If mask/result size mismatches original, the CTD pipeline resized internally —
  always resize mask and final result back to original cv2 dims.
- LaMa inpaint on CPU with HDStrategy.CROP is the quality choice; SimpleLama
  (simple_lama_inpainting) produces more white artifacts on large masks, avoid it.
