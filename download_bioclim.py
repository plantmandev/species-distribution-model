#!/usr/bin/env python3
"""
Download WorldClim 2.1 BioClim data (global, 2.5 arcmin resolution).
Extracts the 8 variables used by sdm.py into bioclim-data/.
"""

import io
import sys
import zipfile
from pathlib import Path

import requests
import shutil

URL     = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"
OUT_DIR = Path("bioclim-data")

# WorldClim filenames → our naming convention
VARIABLES = {
    "wc2.1_2.5m_bio_1.tif":  "bio01_mean.tif",
    "wc2.1_2.5m_bio_4.tif":  "bio04_mean.tif",
    "wc2.1_2.5m_bio_5.tif":  "bio05_mean.tif",
    "wc2.1_2.5m_bio_6.tif":  "bio06_mean.tif",
    "wc2.1_2.5m_bio_12.tif": "bio12_mean.tif",
    "wc2.1_2.5m_bio_15.tif": "bio15_mean.tif",
    "wc2.1_2.5m_bio_18.tif": "bio18_mean.tif",
    "wc2.1_2.5m_bio_19.tif": "bio19_mean.tif",
}


def download_with_progress(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    chunks = []
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total:
            pct = downloaded / total * 100
            mb = downloaded / 1e6
            print(f"\r  {pct:.1f}%  ({mb:.0f} / {total/1e6:.0f} MB)", end="", flush=True)
    print()
    return b"".join(chunks)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    already_done = [v for v in VARIABLES.values() if (OUT_DIR / v).exists()]
    if len(already_done) == len(VARIABLES):
        print("All BioClim files already present, nothing to do.")
        return

    print(f"Downloading WorldClim 2.1 BioClim (2.5 arcmin, global) — ~658 MB...")
    data = download_with_progress(URL)

    print(f"Extracting {len(VARIABLES)} variables...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for src_name, dst_name in VARIABLES.items():
            dst = OUT_DIR / dst_name
            if dst.exists():
                print(f"  skip  {dst_name}")
                continue
            with zf.open(src_name) as src_file, open(dst, "wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file)
            print(f"  done  {dst_name}")

    print(f"\nDone. Files in {OUT_DIR}/:")
    for dst_name in VARIABLES.values():
        p = OUT_DIR / dst_name
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
