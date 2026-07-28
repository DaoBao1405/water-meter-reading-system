from __future__ import annotations

import os

import pandas as pd
import pyodbc


def get_connection() -> pyodbc.Connection:
    server = os.getenv("SQL_SERVER", "192.168.30.251")
    port = os.getenv("SQL_PORT", "1433")
    database = os.getenv("SQL_DATABASE", "Dataloger")
    username = os.getenv("SQL_USERNAME", "sa")
    password = os.getenv("SQL_PASSWORD")
    configured_driver = os.getenv("SQL_DRIVER")

    if not password:
        raise RuntimeError("Missing SQL_PASSWORD. Put it in .env or environment variables.")

    available_drivers = pyodbc.drivers()
    if configured_driver:
        driver = configured_driver
    elif "ODBC Driver 18 for SQL Server" in available_drivers:
        driver = "ODBC Driver 18 for SQL Server"
    elif "ODBC Driver 17 for SQL Server" in available_drivers:
        driver = "ODBC Driver 17 for SQL Server"
    else:
        raise RuntimeError(f"No SQL Server ODBC driver found. Available drivers: {available_drivers}")

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def read_logger_history(conn: pyodbc.Connection, maloger: str, train_days: int) -> pd.DataFrame:
    query = """
    SELECT
        MALOGER,
        NGAYGIO,
        APLUC
    FROM dbo.TBL_APLUC_SAUVAN
    WHERE MALOGER = ?
      AND NGAYGIO >= DATEADD(DAY, -?, GETDATE())
    ORDER BY NGAYGIO
    """
    return pd.read_sql(query, conn, params=[maloger, train_days])


def read_recent_logger_data(conn: pyodbc.Connection, maloger: str, rows_needed: int) -> pd.DataFrame:
    query = f"""
    SELECT TOP ({int(rows_needed)})
        MALOGER,
        NGAYGIO,
        APLUC
    FROM dbo.TBL_APLUC_SAUVAN
    WHERE MALOGER = ?
    ORDER BY NGAYGIO DESC
    """
    df = pd.read_sql(query, conn, params=[maloger])
    return df.sort_values("NGAYGIO").reset_index(drop=True)


def insert_prediction_result(
    conn: pyodbc.Connection,
    maloger: str,
    prediction_time,
    predicted_apluc: float,
    alert_type: str,
    alert_message: str,
) -> None:
    query = """
    INSERT INTO dbo.LSTM_SINGLE_LOGGER_ALERT_RESULT (
        MALOGER,
        PREDICTION_TIME,
        PREDICTED_APLUC,
        ALERT_TYPE,
        ALERT_MESSAGE
    )
    VALUES (?, ?, ?, ?, ?)
    """
    cursor = conn.cursor()
    cursor.execute(
        query,
        maloger,
        prediction_time,
        predicted_apluc,
        alert_type,
        alert_message,
    )
    conn.commit()
