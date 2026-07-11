// clean panel: upload -> /api/clean -> show before/after
const $ = s => document.querySelector(s);
const fileEl = $("#file");
const modelEl = $("#model");
const jpnEl = $("#jpn");
const engineEl = $("#engine");
const dilateEl = $("#dilate");
const dval = $("#dval");
const goBtn = $("#go");
const statusEl = $("#status");
const pvSrc = $("#pvSrc");
const pvDst = $("#pvDst");
const dlBtn = $("#dl");

// panelcleaner ignores these options; ctd-lama uses dilate but not jpn
function syncOpts() {
  const e = engineEl.value;
  jpnEl.disabled = (e !== "lama");
  dilateEl.disabled = (e === "panelcleaner");
  document.getElementById("jpnRow").style.opacity = (e === "lama") ? "1" : ".4";
}
engineEl.onchange = syncOpts;
syncOpts();

dilateEl.oninput = () => { dval.textContent = dilateEl.value; };

goBtn.onclick = async () => {
  const f = fileEl.files[0];
  if (!f) { statusEl.textContent = "Pilih gambar dulu."; statusEl.className = "status err"; return; }
  const engine = engineEl.value;
  statusEl.textContent = "Memproses… (" + (engine === "panelcleaner" ? "PanelCleaner AI" : "LaMa") + ", ~10–30s) ⏳";
  statusEl.className = "status";
  goBtn.disabled = true;
  goBtn.textContent = "Memproses…";
  dlBtn.style.display = "none";
  try {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("engine", engine);
    fd.append("model", modelEl.value);
    fd.append("dilate", dilateEl.value);
    fd.append("jpn", (engine !== "panelcleaner" && jpnEl.checked) ? "true" : "false");
    const r = await fetch("/api/clean", { method: "POST", body: fd });
    const j = await r.json();
    if (!j.ok) {
      statusEl.textContent = "Gagal: " + (j.error || "unknown");
      statusEl.className = "status err";
      return;
    }
    pvSrc.src = j.src + "&t=" + Date.now();
    pvDst.src = j.dst + "&t=" + Date.now();
    dlBtn.href = j.dst + "&t=" + Date.now();
    dlBtn.style.display = "block";
    statusEl.textContent = `Selesai ✓ | model: ${j.model} | mask: ${j.mask_pct}%`;
    statusEl.className = "status ok";
  } catch (e) {
    statusEl.textContent = "Error: " + e.message;
    statusEl.className = "status err";
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Bersihkan ✨";
  }
};
