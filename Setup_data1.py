import requests
from pathlib import Path
import time

# ====================== CONFIG ======================
base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
eeg_dir = Path("data/raw/eeg")
eeg_dir.mkdir(parents=True, exist_ok=True)

runs = ["04", "08", "12"]  # Only required runs


# ====================== DOWNLOAD FUNCTION ======================
def download_file(url, dest_path, max_retries=4):
    dest = Path(dest_path)

    if dest.exists():
        print(f"✅ Already exists: {dest.name}")
        return

    for attempt in range(1, max_retries + 1):
        try:
            print(f"⬇️  {dest.name} (attempt {attempt}/{max_retries})")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(dest, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded: {dest.name}")
            return

        except Exception as e:
            print(f"   Failed: {e}")
            time.sleep(3 * attempt)

    raise RuntimeError(f"Failed after {max_retries} attempts: {url}")


# ====================== MAIN ======================
def main():
    print("=== Downloading EEGMMIDB (ONLY runs 4, 8, 12) ===")

    for subject in range(1, 110):  # S001 to S109
        s = f"S{subject:03d}"

        for run in runs:
            filename = f"{s}R{run}.edf"   # ✅ ONLY .edf
            url = base_url + f"{s}/{filename}"
            dest = eeg_dir / filename

            try:
                download_file(url, str(dest))
            except Exception as e:
                print(f"⚠️ Skipping {filename}: {e}")

    print("✅ EEGMMIDB download finished — only runs 4, 8, 12")


if __name__ == "__main__":
    main()