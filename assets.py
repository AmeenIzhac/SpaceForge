"""One-time download + cache of CC0 photo textures (ambientCG).

`get(name, size)` returns a float32 (size, size, 3) albedo array, loading from
assets/<name>.jpg. Missing assets are downloaded once (zip -> Color map) and
cached; if a download fails the caller falls back to procedural texture, so
the pipeline never breaks offline. Everything here is a one-time cost — at
generation time textures come from the local cache.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).resolve().parent / "assets"
_URL = "https://ambientcg.com/get?file={name}_1K-JPG.zip"
_mem = {}


def _download(name):
    ASSET_DIR.mkdir(exist_ok=True)
    url = _URL.format(name=name)
    req = urllib.request.Request(url, headers={"User-Agent": "spaceforge-sim/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        color = [n for n in z.namelist() if "Color" in n and n.lower().endswith((".jpg", ".png"))]
        if not color:
            raise FileNotFoundError(f"no Color map in {name}")
        img = Image.open(io.BytesIO(z.read(color[0]))).convert("RGB")
    img.save(ASSET_DIR / f"{name}.jpg", quality=92)
    return img


def get(name, size=512):
    """Albedo array for asset `name`, or None if unavailable."""
    key = (name, size)
    if key in _mem:
        return _mem[key]
    path = ASSET_DIR / f"{name}.jpg"
    try:
        if path.exists():
            img = Image.open(path).convert("RGB")
        else:
            img = _download(name)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img, np.float32) / 255.0
    except Exception as e:
        print(f"[assets] {name}: {e} (procedural fallback)")
        arr = None
    _mem[key] = arr
    return arr


# curated CC0 set — walls, floors, ceilings, wood
CATALOG = [
    "Plaster001", "Plaster003",          # painted walls
    "Bricks023", "Bricks066",            # brick walls
    "Concrete034",                       # concrete wall/floor
    "Carpet004", "Carpet008",            # carpets
    "WoodFloor041",                      # parquet
    "Wood051",                           # wood panels / doors
    "Tiles074", "Tiles101",              # floor tiles
    "OfficeCeiling005",                  # ceiling tiles
]


def prefetch():
    ok = []
    for name in CATALOG:
        ok.append((name, get(name) is not None))
    return ok


if __name__ == "__main__":
    for name, good in prefetch():
        print(("ok  " if good else "FAIL"), name)
