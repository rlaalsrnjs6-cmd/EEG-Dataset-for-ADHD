# 목적 피크의 특징을 추출하기전 find_peaks가 제대로 작동하는지 확인
# 기본prominence를 임시 선정


import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#데이터 읽어오기
eeg_data = pd.read_csv("data/raw/adhd_data.csv")
#첫번째 id가져오기
first_id = eeg_data['ID'].iloc[0]
#첫번쨰 id의 데이터 가져오기
first_id_data = eeg_data[eeg_data['ID'] == first_id]

# 데이터를 find_peaks가 읽기 좋은 배열로 반환
signal = first_id_data["Fp1"].to_numpy()
#각각 피크가 300, 500, 1000이상 차이나면 피크로 판단
peaks_300, properties = find_peaks(signal, prominence=300)
peaks_500, properties = find_peaks(signal, prominence=500)
peaks_1000, properties = find_peaks(signal, prominence=1000)

# print("prominence=300")
# print("피크 개수 : " , len(peaks))
# print("처음 20개 피크 위치 : " , peaks[:20])
# print("처음 20개 피크 값 : " , signal[peaks[:20]])
#그래프를 3행으로 한는에 볼수있게 처리
figure, axes = plt.subplots(3,1)
axes[0].plot(range(1000,2000),signal[1000:2000])
axes[0].scatter(peaks_300[(peaks_300 > 1000) & (peaks_300 < 2000)],signal[peaks_300[(peaks_300 > 1000) & (peaks_300 < 2000)]])
axes[1].plot(range(1000,2000),signal[1000:2000])
axes[1].scatter(peaks_500[(peaks_500 > 1000) & (peaks_500 < 2000)],signal[peaks_500[(peaks_500 > 1000) & (peaks_500 < 2000)]])
axes[2].plot(range(1000,2000),signal[1000:2000])
axes[2].scatter(peaks_1000[(peaks_1000 > 1000) & (peaks_1000 < 2000)],signal[peaks_1000[(peaks_1000 > 1000) & (peaks_1000 < 2000)]])
# 피크를 시각적으로 볼수있게 표현
plt.show()