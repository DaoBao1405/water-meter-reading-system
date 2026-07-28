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

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from utils.alert import detect_alert
from utils.db import get_connection, insert_prediction_result, read_recent_logger_data
from utils.features import add_time_features, clean_pressure_data
from utils.model import PressureLSTM


def main() -> None:
    load_dotenv()

    maloger = os.getenv("MALOGER", "0197")
    model_base_dir = Path(os.getenv("MODEL_BASE_DIR", "models"))
    model_dir = model_base_dir / f"logger_{maloger}"

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    feature_scaler = joblib.load(model_dir / "feature_scaler.pkl")
    target_scaler = joblib.load(model_dir / "target_scaler.pkl")

    model = PressureLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(model_dir / "model_lstm.pt", map_location="cpu"))
    model.eval()

    seq_len = int(config["seq_len"])
    feature_cols = config["feature_cols"]
    low_threshold = float(os.getenv("LOW_THRESHOLD", config["low_threshold"]))
    high_threshold = float(os.getenv("HIGH_THRESHOLD", config["high_threshold"]))

    with get_connection() as conn:
        df = read_recent_logger_data(conn, maloger, rows_needed=seq_len + 10)
        df = clean_pressure_data(df)
        df = add_time_features(df)

        if len(df) < seq_len:
            raise RuntimeError(f"Need at least {seq_len} rows. Got {len(df)}.")

        recent = df.tail(seq_len)
        x_scaled = feature_scaler.transform(recent[feature_cols])
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(x_tensor).numpy().reshape(-1, 1)

        predicted_apluc = float(target_scaler.inverse_transform(pred_scaled).ravel()[0])
        prediction_time = df["NGAYGIO"].max() + pd.Timedelta(minutes=15 * int(config["predict_ahead"]))
        alert_type, alert_message = detect_alert(predicted_apluc, low_threshold, high_threshold)

        insert_prediction_result(
            conn=conn,
            maloger=maloger,
            prediction_time=prediction_time.to_pydatetime(),
            predicted_apluc=predicted_apluc,
            alert_type=alert_type,
            alert_message=alert_message,
        )

    print("MALOGER:", maloger)
    print("Prediction time:", prediction_time)
    print("Predicted APLUC:", predicted_apluc)
    print("Alert:", alert_type, "-", alert_message)


if __name__ == "__main__":
    main()
