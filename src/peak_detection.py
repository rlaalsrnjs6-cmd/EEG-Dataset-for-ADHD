import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

eeg_data = pd.read_csv("data/raw/adhd_data.csv")

first_id = eeg_data['ID'].iloc[0]

first_id_data = eeg_data[eeg_data['ID'] == first_id]

# 배열로 변경
signal = first_id_data["Fp1"].to_numpy()

peaks_300, properties = find_peaks(signal, prominence=300)
peaks_500, properties = find_peaks(signal, prominence=500)
peaks_1000, properties = find_peaks(signal, prominence=1000)

# print("prominence=300")
# print("피크 개수 : " , len(peaks))
# print("처음 20개 피크 위치 : " , peaks[:20])
# print("처음 20개 피크 값 : " , signal[peaks[:20]])

figure, axes = plt.subplots(3,1)
axes[0].plot(range(1000,2000),signal[1000:2000])
axes[0].scatter(peaks_300[(peaks_300 > 1000) & (peaks_300 < 2000)],signal[peaks_300[(peaks_300 > 1000) & (peaks_300 < 2000)]])
axes[1].plot(range(1000,2000),signal[1000:2000])
axes[1].scatter(peaks_500[(peaks_500 > 1000) & (peaks_500 < 2000)],signal[peaks_500[(peaks_500 > 1000) & (peaks_500 < 2000)]])
axes[2].plot(range(1000,2000),signal[1000:2000])
axes[2].scatter(peaks_1000[(peaks_1000 > 1000) & (peaks_1000 < 2000)],signal[peaks_1000[(peaks_1000 > 1000) & (peaks_1000 < 2000)]])
plt.show()