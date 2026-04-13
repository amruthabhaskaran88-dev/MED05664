# setup_data.py  ← UPDATED WITH FALLBACK + 502 HANDLING
import os
import zipfile
import json
from datetime import datetime
from pathlib import Path

import requests
from tqdm import tqdm

def resumable_download(url: str, dest_path: str, max_retries=5):
    dest = Path(dest_path)
    if dest.exists():
        print(f"✅ {dest.name} already exists")
        return

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for attempt in range(1, max_retries + 1):
        print(f"⬇️  Attempt {attempt}/{max_retries}: {dest.name}")
        try:
            response = requests.get(url, stream=True, timeout=(30, 1800), headers=headers)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                desc=dest.name, total=total_size, unit="iB", unit_scale=True
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"✅ {dest.name} downloaded successfully")
            return

        except requests.exceptions.HTTPError as e:
            if response.status_code == 502:
                print("   502 Bad Gateway → retrying...")
            else:
                print(f"   Error: {e}")
            if attempt == max_retries:
                raise
            time.sleep(10 * attempt)  # longer wait on failure

    raise RuntimeError(f"Failed to download after {max_retries} attempts: {url}")

def main():
    project_root = Path.cwd()
    raw = project_root / "data" / "raw"

    (raw / "PAMAP2").mkdir(parents=True, exist_ok=True)
    (raw / "WISDM").mkdir(parents=True, exist_ok=True)
    (raw / "EEGMMIDB").mkdir(parents=True, exist_ok=True)
    (raw / "PTB-XL").mkdir(parents=True, exist_ok=True)

    # 1. PAMAP2 (with fallback link)
    print("\n=== Downloading PAMAP2 ===")
    pamap_url1 = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
    pamap_url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/pamap2+physical+activity+monitoring.zip"
    try:
        resumable_download(pamap_url1, str(raw / "PAMAP2.zip"))
    except:
        print("Primary link failed → trying fallback link...")
        resumable_download(pamap_url2, str(raw / "PAMAP2.zip"))

    with zipfile.ZipFile(raw / "PAMAP2.zip") as z:
        z.extractall(raw / "PAMAP2")
    (raw / "PAMAP2.zip").unlink(missing_ok=True)

    # 2. WISDM
    print("\n=== Downloading WISDM ===")
    resumable_download(
        "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
        str(raw / "WISDM.zip")
    )
    with zipfile.ZipFile(raw / "WISDM.zip") as z:
        z.extractall(raw / "WISDM")
    (raw / "WISDM.zip").unlink(missing_ok=True)

    # 3. EEGMMIDB full ZIP (we filter to runs 4,8,12 later)
    print("\n=== Downloading EEGMMIDB (we will only use runs 4, 8, 12) ===")
    resumable_download(
        "https://physionet.org/content/eegmmidb/get-zip/1.0.0/",
        str(raw / "EEGMMIDB.zip")
    )
    with zipfile.ZipFile(raw / "EEGMMIDB.zip") as z:
        z.extractall(raw / "EEGMMIDB")
    (raw / "EEGMMIDB.zip").unlink(missing_ok=True)

    # 4. PTB-XL
    print("\n=== Downloading PTB-XL ===")
    resumable_download(
        "https://physionet.org/content/ptb-xl/get-zip/1.0.3/",
        str(raw / "PTB-XL.zip")
    )
    with zipfile.ZipFile(raw / "PTB-XL.zip") as z:
        z.extractall(raw / "PTB-XL")
    (raw / "PTB-XL.zip").unlink(missing_ok=True)

    # 5. Manifest
    print("\n=== Creating manifest.json ===")
    manifest = {
        "download_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "EEGMMIDB full ZIP downloaded – preprocessing will use ONLY runs 4, 8, 12 as required",
        "datasets": {
            "PAMAP2": {"url": pamap_url1, "version": "UCI"},
            "WISDM": {"url": "https://archive.ics.uci.edu/dataset/507", "version": "UCI 2019"},
            "EEGMMIDB": {"url": "https://physionet.org/content/eegmmidb/1.0.0/", "version": "1.0.0"},
            "PTB_XL": {"url": "https://physionet.org/content/ptb-xl/1.0.3/", "version": "1.0.3"},
        }
    }
    with open(raw / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n🎉 SETUP COMPLETE!")
    print("All datasets are ready.")
    print("EEGMMIDB will be filtered to runs 4, 8, and 12 in preprocessing.")

if __name__ == "__main__":
    main()