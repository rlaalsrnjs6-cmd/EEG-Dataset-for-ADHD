from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
 
def validate_eeg(data, eeg_channels, input_hash,  code_hash):

    required_columns = eeg_channels + ["ID", "Class"]
    missing_columns  = []
    channels = []

    for channel in required_columns:
        if channel not in data.columns:
            missing_columns .append(channel)

    if missing_columns :
        save_error_log(
            code_hash,
            input_hash,
            "missing_channels_error",
            f"필수 EEG 컬럼이 없습니다 : {missing_columns }"
            )
        raise ValueError(f"필수 EEG 컬럼이 없습니다 : {missing_columns }")

    if data[eeg_channels].isnull().any().any():
        save_error_log(
            code_hash,
            input_hash,
            "isnull_eeg",
            "EEG데이터에 결측치가 있습니다."
            )
        raise ValueError("EEG데이터에 결측치가 있습니다.")

    for channel in eeg_channels:
        if not pd.api.types.is_numeric_dtype(data[channel]):
            channels.append(channel)
    if channels:
        save_error_log(
            code_hash,
            input_hash,
            "non_numeric_eeg",
            f"EEG 채널에 숫자가 아닌 값이 있습니다 : {channels}"
        )
        raise ValueError(f"EEG 채널에 숫자가 아닌 값이 있습니다 : {channels}")
    
    eeg_zero = data[eeg_channels].eq(0).all(axis=1)

    if eeg_zero.any():
        save_error_log(
            code_hash,
            input_hash,
            "all_zero_eeg",
            "EEG 채널이 모두 0인 행이 있습니다."
        )
        raise ValueError("EEG 채널이 모두 0인 행이 있습니다.")


    if len(data) < 128:
        save_error_log(
            code_hash,
            input_hash,
            "insufficient_windowData_error",
            "EEG데이터가 128행보다 적습니다."
        )
        raise ValueError("EEG데이터가 128행보다 적습니다.")


def save_error_log(code_hash, input_hash, error_type, message):
    error_record ={
        "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "code_hash": code_hash,
        "input_hash": input_hash,
        "error_type": error_type,
        "message": message
    }

    log_path = Path("logs/error_log.csv")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_df = pd.DataFrame([error_record])

    log_df.to_csv(
        log_path,
        mode="a",
        header=not log_path.exists(),
        index=False,
        encoding="utf-8-sig"
    )
