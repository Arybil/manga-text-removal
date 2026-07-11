import os, hashlib, asyncio, json
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = "/root/manga-web"
STATIC = os.path.join(ROOT, "static")

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    # the whole site is the clean-manga tool
    return FileResponse(os.path.join(STATIC, "clean.html"))


@app.get("/clean")
async def clean_page():
    return FileResponse(os.path.join(STATIC, "clean.html"))


CLEAN_DIR = os.path.join(ROOT, "cache", "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)
CLEAN_SCRIPT = "/root/imgtools/clean_manga_lama.py"
CLEAN_PY = "/root/imgtools/bin/python"
PCLEANER_SCRIPT = "/root/imgtools/pclean_cli.py"
PCLEANER_PY = "/root/pcleaner-venv/bin/python"
CTD_SCRIPT = "/root/imgtools/ctd_lama.py"
CTD_PY = "/root/imgtools/bin/python"


@app.post("/api/clean")
async def api_clean(file: UploadFile = None, model: str = Form("lama"), dilate: int = Form(10),
                    jpn: bool = Form(False), engine: str = Form("ctd-lama")):
    if file is None:
        return JSONResponse({"ok": False, "error": "no file"}, status_code=400)
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        return JSONResponse({"ok": False, "error": "format harus png/jpg/webp"}, status_code=400)
    try:
        data = await file.read()
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"read failed: {e}"}, status_code=400)
    if not data:
        return JSONResponse({"ok": False, "error": "file kosong"}, status_code=400)
    h = hashlib.md5(data).hexdigest()
    src = os.path.join(CLEAN_DIR, h + "_src.png")
    dst = os.path.join(CLEAN_DIR, h + "_clean.png")
    with open(src, "wb") as f:
        f.write(data)

    if engine == "panelcleaner":
        cmd = [PCLEANER_PY, PCLEANER_SCRIPT, src, dst, "--device", "cpu"]
        timeout_s = 300
    elif engine == "ctd-lama":
        cmd = [CTD_PY, CTD_SCRIPT, src, dst, "--device", "cpu", "--dilate", str(dilate)]
        timeout_s = 300
    else:
        model = model if model in ("lama", "manga") else "lama"
        cmd = [CLEAN_PY, CLEAN_SCRIPT, src, dst, "--model", model,
               "--dilate", str(dilate)] + (["--jpn"] if jpn else [])
        timeout_s = 180

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return JSONResponse({"ok": False, "error": f"timeout (>{timeout_s}s)"}, status_code=504)
    if proc.returncode != 0 or not os.path.exists(dst):
        return JSONResponse({"ok": False, "error": (stderr or stdout).decode()[-800:]}, status_code=500)
    # extract json result from stdout (last valid JSON line)
    res = None
    for line in stdout.decode().strip().splitlines():
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                res = json.loads(s)
            except Exception:
                pass
    return JSONResponse({
        "ok": True,
        "engine": engine,
        "model": model if engine != "panelcleaner" else "panelcleaner",
        "mask_pct": res.get("mask_pct") if res else None,
        "src": "/api/clean/file?name=" + os.path.basename(src),
        "dst": "/api/clean/file?name=" + os.path.basename(dst),
    })


@app.get("/api/clean/file")
async def api_clean_file(name: str):
    # prevent path traversal
    name = os.path.basename(name)
    p = os.path.join(CLEAN_DIR, name)
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)
