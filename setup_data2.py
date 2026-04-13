import requests
from pathlib import Path

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"
ECG_DIR = Path("data/raw/ecg")
ECG_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest):
    if dest.exists():
        print(f"✅ Exists: {dest.name}")
        return

    print(f"⬇️ Downloading {dest.name}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(dest, "wb") as f:
        f.write(r.content)

    print(f"✅ Downloaded {dest.name}")


def download_ecg():
    print("🚀 Downloading ECG metadata + sample signals")

    # metadata
    metadata_files = [
        "ptbxl_database.csv",
        "scp_statements.csv"
    ]

    for file in metadata_files:
        download_file(BASE_URL + file, ECG_DIR / file)

    # sample ECG signals
    signal_dir = ECG_DIR / "records100" / "00000"
    signal_dir.mkdir(parents=True, exist_ok=True)

    sample_records = [f"{i:05d}_lr" for i in range(1, 101)]    

    for record in sample_records:
        for ext in [".hea", ".dat"]:
            filename = record + ext
            url = BASE_URL + f"records100/00000/{filename}"
            dest = signal_dir / filename
            download_file(url, dest)

    print("🎉 ECG metadata + sample signal files ready")


def main():
    download_ecg()


if __name__ == "__main__":
    main()