from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from utils.db import get_connection, read_logger_history
from utils.features import add_time_features, chronological_split, clean_pressure_data, make_windows
from utils.model import PressureLSTM


def train_model(model, train_loader, val_x, val_y, epochs, lr, patience, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val = float("inf")
    wait = 0
    history = []

    val_x_t = torch.tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.tensor(val_y, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x_t), val_y_t).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, x, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def main() -> None:
    load_dotenv()

    maloger = os.getenv("MALOGER", "0197")
    train_days = int(os.getenv("TRAIN_DAYS", "90"))
    seq_len = int(os.getenv("SEQ_LEN", "48"))
    predict_ahead = int(os.getenv("PREDICT_AHEAD", "1"))
    model_base_dir = Path(os.getenv("MODEL_BASE_DIR", "models"))
    model_dir = model_base_dir / f"logger_{maloger}"
    model_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(os.getenv("EPOCHS", "50"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
    patience = int(os.getenv("PATIENCE", "7"))
    hidden_size = int(os.getenv("HIDDEN_SIZE", "24"))
    num_layers = int(os.getenv("NUM_LAYERS", "1"))
    dropout = float(os.getenv("DROPOUT", "0.0"))

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Training logger {maloger} with last {train_days} days")
    train_csv_path = os.getenv("TRAIN_CSV_PATH")
    if train_csv_path:
        print(f"Reading training data from CSV: {train_csv_path}")
        raw_df = pd.read_csv(train_csv_path)
        if "MALOGER" in raw_df.columns:
            raw_df["MALOGER"] = raw_df["MALOGER"].astype(str).str.zfill(4)
            raw_df = raw_df[raw_df["MALOGER"] == maloger].copy()
    else:
        with get_connection() as conn:
            raw_df = read_logger_history(conn, maloger, train_days)

    if raw_df.empty:
        raise RuntimeError(f"No data found for MALOGER={maloger}")

    df = clean_pressure_data(raw_df)
    df = add_time_features(df)
    df["target"] = df["APLUC"].shift(-predict_ahead)
    df["target_time"] = df["NGAYGIO"].shift(-predict_ahead)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["APLUC", "sin_day", "cos_day", "sin_week", "cos_week"]
    train_raw_end = int(len(df) * 0.70)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(df.loc[: train_raw_end - 1, feature_cols])
    target_scaler.fit(df.loc[: train_raw_end - 1, ["target"]])

    scaled_features = feature_scaler.transform(df[feature_cols])
    scaled_target = target_scaler.transform(df[["target"]]).ravel()
    x, y, ts = make_windows(scaled_features, scaled_target, df["target_time"].to_numpy(), seq_len)

    if len(x) < 100:
        raise RuntimeError(f"Not enough windows to train. Got {len(x)} windows.")

    train_slice, val_slice, test_slice = chronological_split(len(x))
    x_train, y_train = x[train_slice], y[train_slice]
    x_val, y_val = x[val_slice], y[val_slice]
    x_test, y_test = x[test_slice], y[test_slice]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Rows:", len(df), "Windows:", len(x), "Train/Val/Test:", len(x_train), len(x_val), len(x_test))

    model = PressureLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    history = train_model(model, train_loader, x_val, y_val, epochs, learning_rate, patience, device)

    pred_test_scaled = predict(model, x_test, device)
    test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    test_pred = target_scaler.inverse_transform(pred_test_scaled.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(test_actual, test_pred))
    rmse = float(mean_squared_error(test_actual, test_pred) ** 0.5)
    q25 = float(df["APLUC"].quantile(0.25))
    q75 = float(df["APLUC"].quantile(0.75))

    config = {
        "maloger": maloger,
        "train_days": train_days,
        "seq_len": seq_len,
        "predict_ahead": predict_ahead,
        "feature_cols": feature_cols,
        "input_size": len(feature_cols),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "low_threshold": float(os.getenv("LOW_THRESHOLD", q25)),
        "high_threshold": float(os.getenv("HIGH_THRESHOLD", q75)),
        "data_q25": q25,
        "data_q75": q75,
        "test_mae": mae,
        "test_rmse": rmse,
    }

    torch.save(model.state_dict(), model_dir / "model_lstm.pt")
    joblib.dump(feature_scaler, model_dir / "feature_scaler.pkl")
    joblib.dump(target_scaler, model_dir / "target_scaler.pkl")
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    test_results = pd.DataFrame(
        {
            "NGAYGIO": pd.to_datetime(ts[test_slice]),
            "actual_apluc": test_actual,
            "predicted_apluc": test_pred,
            "abs_error": np.abs(test_actual - test_pred),
        }
    )
    test_results.to_csv(model_dir / "test_results.csv", index=False, encoding="utf-8-sig")

    print("Saved model artifacts to:", model_dir.resolve())
    print(json.dumps(config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
