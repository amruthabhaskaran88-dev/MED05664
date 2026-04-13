import pandas as pd
import numpy as np
from pathlib import Path

# ================= CONFIG =================
TARGET_HZ = 20
PRETRAIN_SIZE = 200   # 10 sec
SUP_SIZE = 100        # 5 sec
SUP_STEP = 50         # 50% overlap

CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# ================= LABEL MAP =================
LABEL_MAP = {
    "1": "walking",
    "2": "running",
    "3": "cycling",
    "4": "sedentary"
}

# ================= RESAMPLE =================
def resample_df(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp")
    df = df.resample("50ms").mean().interpolate()  # 20 Hz
    return df.reset_index()

# ================= CLEAN =================
def clean(df):
    df = df.dropna()
    df[CHANNELS] = (df[CHANNELS] - df[CHANNELS].mean()) / df[CHANNELS].std()
    return df

# ================= WINDOW =================
def create_windows(data, labels=None, size=100, step=50):
    X, y = [], []

    for i in range(0, len(data) - size + 1, step):
        X.append(data[i:i+size])

        if labels is not None:
            lab = labels[i:i+size]
            y.append(max(set(lab), key=list(lab).count))

    return np.array(X), np.array(y) if labels is not None else None

# ================= PAMAP2 =================
def load_pamap2_all():
    pamap_dir = Path("data/raw/PAMAP2/PAMAP2_Dataset/Protocol")
    all_df = []

    for file in pamap_dir.glob("*.dat"):
        subject = file.stem

        df = pd.read_csv(file, sep='\s+', header=None)

        df = df.iloc[:, [0, 1, 4, 5, 6, 10, 11, 12]]
        df.columns = ["timestamp", "activity"] + CHANNELS

        df = df[df["activity"] != 0]

        df["label"] = df["activity"].astype(str)
        df["subject_id"] = subject
        df["dataset"] = "PAMAP2"

        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

# ================= WISDM =================
def load_wisdm():
    file = Path("data/raw/WISDM/wisdm.csv")  # adjust if needed

    df = pd.read_csv(file)

    df.columns = ["user", "activity", "timestamp"] + CHANNELS

    df["label"] = df["activity"].str.lower()
    df["subject_id"] = df["user"]
    df["dataset"] = "WISDM"

    return df

# ================= MAIN =================
def main():
    print("🚀 Loading datasets...")

    pamap = load_pamap2_all()
    wisdm = load_wisdm()

    print("PAMAP2:", pamap.shape)
    print("WISDM:", wisdm.shape)

    # Combine
    df = pd.concat([pamap, wisdm], ignore_index=True)

    print("Combined:", df.shape)

    # Process
    df = resample_df(df)
    df = clean(df)

    signals = df[CHANNELS].values
    labels = df["label"].values

    # Pretraining
    X_pre, _ = create_windows(signals, None, PRETRAIN_SIZE, PRETRAIN_SIZE)

    # Supervised
    X, y = create_windows(signals, labels, SUP_SIZE, SUP_STEP)

    print("Pretrain:", X_pre.shape)
    print("Supervised:", X.shape)

    # Save
    out = Path("data/processed/har")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_pretrain.npy", X_pre)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)

    print("🎉 HAR dataset ready!")


if __name__ == "__main__":
    main()