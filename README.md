# manga-text-removal

🌐 **Live web panel:** https://komikaku.duckdns.org/clean
(upload a manga page, pick an engine, download the cleaned art — `ctd-lama` is the default)

Strip **all** text from manga panels (speech bubbles, narration boxes, caption/info
bars, vertical text) for re-lettering. Three interchangeable engines, all inpaint
via [LaMa](https://github.com/Sanster/lama-cleaner):

| Engine | What it removes | Verified result (1024x722 page) |
|---|---|---|
| **`ctd_lama.py`** ⭐ default | everything (AI detection) | 0 text left, 2.57% white artifact |
| `clean_manga_lama.py` | everything (Tesseract OCR) | 0 text left, 3.22% white artifact |
| `pclean_cli.py` | speech bubbles only | bubbles clean, boxes/vertical escape |

`ctd_lama.py` is recommended: it reuses PanelCleaner's
[comic-text-detector](https://github.com/VoxelCubes/PanelCleaner) (a YOLO model)
to build a precise text mask, then inpaints with LaMa. This catches narration
boxes and caption bars that PanelCleaner's own pipeline skips.

## Quick start

```bash
# 1. create the two venvs
python3 -m venv pcleaner-venv && pcleaner-venv/bin/pip install "pcleaner==2.11.11" "transformers<5"
python3 -m venv imgtools &&      imgtools/bin/pip install lama-cleaner opencv-python-headless pytesseract pillow numpy

# 2. run (first run downloads the CTD model ~tens of MB)
imgtools/bin/python scripts/ctd_lama.py input.png cleaned.png --dilate 8
# -> {"ok": true, "engine": "ctd-lama", "mask_pct": 7.6, "dst": "cleaned.png"}
```

The scripts read the venv locations from env vars (`PCLEANER_VENV`, `IMGTOOLS_VENV`,
`CTD_MODEL`, `PCLEANER_BIN`) and fall back to `/root/...` defaults if unset.

## Scripts

- `scripts/ctd_lama.py` — **primary engine** (AI text detection + LaMa inpaint)
- `scripts/clean_manga_lama.py` — legacy Tesseract OCR engine (`--model lama|manga`, `--jpn`)
- `scripts/pclean_cli.py` — PanelCleaner wrapper (bubbles only)

## Web panel

A FastAPI endpoint (`POST /api/clean`) that accepts an upload + `engine`/`model`/
`dilate`/`jpn` form fields, runs the matching script, and returns JSON. See `SKILL.md`
for the integration snippet (use `Form(...)` + `asyncio.create_subprocess_exec`).

## Verify a result

VLM/vision checks hallucinate text in clean bubbles — use OCR as ground truth:

```bash
python3 - <<'PY'
import cv2, pytesseract, numpy as np
def ocr(p):
    img = cv2.imread(p); txt=[]
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
print("LEFTOVER LEGIT:", [w for w in ocr("clean.png") if w not in ("A","I","THE")])
clean=cv2.imread("clean.png"); g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY)
print("WHITE ARTIFACT: %.2f%%" % (100*(g>245).mean()))
PY
```

Target: `LEFTOVER LEGIT: []`, white artifact < 4%.

## License

MIT — use freely, attribute if you like.
