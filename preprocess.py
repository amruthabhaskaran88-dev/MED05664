# =========================================
# FINAL MULTIMODAL PIPELINE (HAR + EEG + ECG)
# =========================================

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ================= CONFIG =================
BASE = Path("data")
RAW = BASE / "raw"
OUT = BASE / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

# =========================================
# ============ HAR ========================
# =========================================

def load_pamap2():
    path = RAW / "PAMAP2/PAMAP2_Dataset/Protocol"
    dfs = []

    for f in path.glob("*.dat"):
        df = pd.read_csv(f, sep=" ", header=None)

        df = df.iloc[:, [0,1,4,5,6,10,11,12]]
        df.columns = ["timestamp","activity","acc_x","acc_y","acc_z",
                      "gyro_x","gyro_y","gyro_z"]

        df["subject_id"] = f.stem
        df["dataset"] = "PAMAP2"

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_wisdm():
    acc_file = RAW / "WISDM/wisdm-dataset/raw/watch/accel/data_1600_accel_watch.txt"
    gyro_file = RAW / "WISDM/wisdm-dataset/raw/watch/gyro/data_1600_gyro_watch.txt"

    acc = pd.read_csv(acc_file, header=None)
    gyro = pd.read_csv(gyro_file, header=None)

    acc.columns = ["user","activity","timestamp","acc_x","acc_y","acc_z"]
    gyro.columns = ["user","activity","timestamp","gyro_x","gyro_y","gyro_z"]

    df = pd.merge(acc, gyro, on=["user","timestamp"], how="inner")

    df["activity"] = df["activity_x"]
    df = df.drop(columns=["activity_x","activity_y"])

    df["subject_id"] = df["user"]
    df["dataset"] = "WISDM"

    return df


def clean(df):
    df = df.copy()

    for col in CHANNELS:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(";", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=CHANNELS)

    df.loc[:, CHANNELS] = df[CHANNELS].clip(-20, 20)

    return df


def downsample(df, factor=5):
    return df.iloc[::factor].reset_index(drop=True)


def create_windows(df, size=100, step=50):
    X, y = [], []

    if len(df) < size:
        return np.array([]), np.array([])

    data = df[CHANNELS].values
    labels = df["activity"].values

    for i in range(0, len(df) - size, step):
        win = data[i:i+size]

        if np.isnan(win).any():
            continue

        X.append(win.T.astype(np.float32))
        y.append(pd.Series(labels[i:i+size]).mode()[0])

    return np.array(X), np.array(y)


def process_har():
    print("🚀 HAR processing...")

    pamap = load_pamap2()
    wisdm = load_wisdm()

    all_X, all_y, meta = [], [], []

    for df in [pamap, wisdm]:

        print("👉 Dataset:", df["dataset"].iloc[0])

        df = clean(df)
        df = downsample(df)

        X, y = create_windows(df)

        print("Windows:", len(X))

        if len(X) == 0:
            continue

        for i in range(len(X)):
            meta.append({
                "sample_id": len(meta),
                "dataset_name": df["dataset"].iloc[0],
                "modality": "HAR",
                "subject_or_patient_id": str(df["subject_id"].iloc[0]),
                "label_or_event": str(y[i]),
                "sampling_rate_hz": 20,
                "n_channels": 6,
                "n_samples": 100,
                "channel_schema": "6ch_accel+gyro",
                "qc_flags": "ok"
            })

        all_X.append(X)
        all_y.append(y)

    if len(all_X) == 0:
        raise ValueError("❌ HAR FAILED")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    meta = pd.DataFrame(meta)

    np.save(OUT / "har_X.npy", X)
    np.save(OUT / "har_y.npy", y)
    meta.to_csv(OUT / "har_metadata.csv", index=False)

    print("✅ HAR done:", X.shape)


# =========================================
# ============ EEG (FIXED) ================
# =========================================

def process_eeg():
    print("🧠 EEG processing...")

    import mne

    eeg_dir = RAW / "EEGMMIDB/eeg"

    TARGET_FS = 160
    TARGET_LEN = 640  # 4 sec

    X_all, meta = [], []

    for file in eeg_dir.glob("*.edf"):

        if not any(r in file.name for r in ["R04","R08","R12"]):
            continue

        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)

        raw.resample(TARGET_FS)

        events, _ = mne.events_from_annotations(raw)

        for ev in events:
            if ev[2] not in [2,3]:
                continue

            start = ev[0]
            end = start + TARGET_LEN

            if end > raw.n_times:
                continue

            data = raw.get_data(start=start, stop=end)

            if data.shape[1] != TARGET_LEN:
                continue

            X_all.append(data.astype(np.float32))

            meta.append({
                "sample_id": len(X_all),
                "dataset_name": "EEGMMIDB",
                "modality": "EEG",
                "subject_or_patient_id": file.name[:4],
                "source_file_or_record": file.name,
                "label_or_event": "T1" if ev[2]==2 else "T2",
                "sampling_rate_hz": TARGET_FS,
                "n_channels": data.shape[0],
                "n_samples": TARGET_LEN,
                "channel_schema": "EEG_64",
                "qc_flags": "ok"
            })

    if len(X_all) == 0:
        print("⚠️ No EEG extracted")
        return

    X = np.stack(X_all)

    np.save(OUT / "eeg_X.npy", X)
    pd.DataFrame(meta).to_csv(OUT / "eeg_metadata.csv", index=False)

    print("✅ EEG done:", X.shape)


# =========================================
# ============ ECG ========================
# =========================================

def process_ecg():
    print("❤️ ECG processing...")

    import wfdb

    path = RAW / "PTB-XL"

    df = pd.read_csv(path / "ptbxl_database.csv")
    df["scp_codes"] = df["scp_codes"].apply(eval)

    X_all, meta = [], []

    for i, row in df.iterrows():

        record = row["filename_lr"]

        signal, _ = wfdb.rdsamp(str(path / record))
        signal = signal.T.astype(np.float32)

        X_all.append(signal)

        meta.append({
            "sample_id": i,
            "dataset_name": "PTB-XL",
            "modality": "ECG",
            "subject_or_patient_id": row["patient_id"],
            "source_file_or_record": record,
            "label_or_event": str(row["scp_codes"]),
            "sampling_rate_hz": 100,
            "n_channels": signal.shape[0],
            "n_samples": signal.shape[1],
            "channel_schema": "12lead",
            "qc_flags": "ok"
        })

    X = np.array(X_all)

    np.save(OUT / "ecg_X.npy", X)
    pd.DataFrame(meta).to_csv(OUT / "ecg_metadata.csv", index=False)

    print("✅ ECG done:", X.shape)


# =========================================
# ============ MANIFEST ===================
# =========================================

def create_manifest():
    manifest = []

    for f in OUT.glob("*"):
        entry = {"file": str(f), "size_bytes": f.stat().st_size}

        if f.suffix == ".npy":
            arr = np.load(f, mmap_mode="r")
            entry["shape"] = list(arr.shape)

        elif f.suffix == ".csv":
            df = pd.read_csv(f)
            entry["rows"] = len(df)

        manifest.append(entry)

    with open(OUT / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("📦 Manifest created")


# =========================================
# ============ MAIN =======================
# =========================================

def main():
    process_har()
    process_eeg()
    process_ecg()
    create_manifest()

    print("\n🎉 ALL DONE — SUBMISSION READY")


if __name__ == "__main__":
    main()