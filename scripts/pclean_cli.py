#!/usr/bin/env python3
"""PanelCleaner (pcleaner) wrapper for the manga text-removal toolkit.

Usage: pclean_cli.py <src> <dst> [--device cpu]
Runs pcleaner on a single image, copies the cleaned output to <dst>,
prints JSON result. Model data auto-downloads on first run (~500MB).

NOTE: PanelCleaner only removes text INSIDE speech bubbles (it intersects the
text mask with the bubble-outline mask). Narration boxes, caption bars, and
vertical text outside bubbles are NOT removed. For full removal use ctd_lama.py.

The pcleaner binary is expected on PATH (or in the same venv as this python).
Override with the PCLEANER_BIN env var.
"""
import sys, os, json, argparse, tempfile, shutil, subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    src = args.src
    if not os.path.exists(src):
        print(json.dumps({"ok": False, "error": f"no src {src}"}))
        sys.exit(1)

    # pcleaner writes to <dir>/cleaned/<name>_cleaned.ext  (older) or
    # <dir>/<name>_clean.<ext> (newer); search both.
    workdir = tempfile.mkdtemp(prefix="pcl_")
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    shutil.copy(src, os.path.join(workdir, base))

    env = dict(os.environ)
    env["PCLEANER_DEVICE"] = args.device
    pcleaner_bin = os.environ.get("PCLEANER_BIN")
    if not pcleaner_bin:
        pcleaner_bin = os.path.join(os.path.dirname(sys.executable), "pcleaner")
    cmd = [pcleaner_bin, "clean", "--save-only-cleaned",
           "--output_dir", workdir, os.path.join(workdir, base)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=workdir, env=env, timeout=300)
    except subprocess.TimeoutExpired:
        print(json.dumps({"ok": False, "error": "timeout (>300s)"}))
        sys.exit(1)

    if proc.returncode != 0:
        print(json.dumps({"ok": False, "error": (proc.stderr or proc.stdout)[-1000:]}))
        sys.exit(1)

    # find cleaned output: <name>_clean.<ext> in workdir or workdir/cleaned
    out = None
    for d in (workdir, os.path.join(workdir, "cleaned")):
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith(name + "_clean") and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                out = os.path.join(d, f)
                break
        if out:
            break

    if out is None:
        print(json.dumps({"ok": False, "error": "no cleaned output produced"}))
        sys.exit(1)

    shutil.copy(out, args.dst)

    # mask coverage estimate
    try:
        import cv2
        img = cv2.imread(src)
        res = cv2.imread(args.dst)
        total = img.shape[0] * img.shape[1]
        diff = (cv2.absdiff(img, res) > 30).sum() / 3
        mask_pct = round(100 * diff / total, 1)
    except Exception:
        mask_pct = None

    print(json.dumps({
        "ok": True,
        "engine": "panelcleaner",
        "mask_pct": mask_pct,
        "dst": args.dst,
    }))


if __name__ == "__main__":
    main()
