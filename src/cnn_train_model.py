import pandas as pd
import torch
import torch.nn as nn

raw_df = pd.read_csv("data/raw/adhd_data.csv")

window_size = 128
eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']

#중복제거
drop_data = raw_df.drop_duplicates()
# print(len(raw_df))
# print(len(drop_data))

null_data = drop_data.dropna()
# print(len(null_data))

zero_data = null_data[eeg_channels].eq(0).all(axis=1)
eeg_data = null_data[~zero_data]
# print(zero_data.sum())
# print(len(eeg_data))


for current_id in eeg_data["ID"].unique():
    id_data = eeg_data[eeg_data["ID"]==current_id]
    for start in range(0, len(id_data), window_size):
        window_data = id_data.iloc[start: start+window_size]
        if len(window_data) < window_size:
            continue
        window_eeg = window_data[eeg_channels]
        window_eeg = window_eeg.T
print(window_eeg.shape)

conv1 = nn.Conv1d(
    in_channels=10,
    out_channels=32,
    kernel_size=3
)

#넘파이 배열로 바꾸고 그값을 파이토치 텐서로 바꾸면서 숫자타입을 float32로 저장 32비트
window_tensor = torch.tensor(
    window_eeg.to_numpy(),
    dtype=torch.float32
)