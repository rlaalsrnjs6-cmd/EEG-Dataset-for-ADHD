import pandas as pd
 
def validate_eeg(data, eeg_channels):
    missing_channels = []

    for channel in eeg_channels:
        if channel not in data.columns:
            missing_channels.append(channel)

    if missing_channels:
        raise ValueError(f"필수 EEG 채널이 없습니다 : {missing_channels}")

    if data[eeg_channels].isnull().any().any():
        raise ValueError("EEG데이터에 결측치가 있습니다.")

    for channel in eeg_channels:
        if not pd.api.types.is_numeric_dtype(data[channel]):
            raise ValueError(f"EEG 채널에 숫자가 아닌 값이 있습니다 : {channel}")
    
    eeg_zero = data[eeg_channels].eq(0).all(axis=1)

    if eeg_zero.any():
        raise ValueError("EEG 채널이 모두 0인 행이 있습니다.")


    if len(data) < 128:
        raise ValueError("EEG데이터가 128행보다 적습니다.")
