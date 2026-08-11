import pandas as pd
from scipy.signal import find_peaks

def extract_features(window, eeg_channels):
     #128행의 채널값만 가져오기
    window_eeg = window[eeg_channels]   
    # print(window_eeg.shape)
    #128행의 컬럼별 특징 하나로 추출 기본이 axis=0(세로)
    window_mean = window_eeg.mean()
    # print(window_mean)
    window_std = window_eeg.std()
    # print(window_std)
    window_min = window_eeg.min()
    # print(window_min)
    window_max = window_eeg.max()
    # print(window_max)1
    #이름이 같아지지않게 구분
    window_mean.index = window_mean.index + "_mean"
    window_std.index = window_std.index + "_std"
    window_min.index = window_min.index + "_min"
    window_max.index = window_max.index + "_max"
    window_peak_feature = pd.Series(dtype=float)
    #피크 특징 더하기
    for channel in eeg_channels:
        #128행의 채널 데이터를 배열로 만들기
        channel_signal = window[channel].to_numpy()
        window_peak, window_properties = find_peaks(channel_signal,prominence=200)
        window_peak_data = channel_signal[window_peak]
        window_peak_count = len(window_peak)
        if window_peak_count > 0:
            window_peak_mean = window_peak_data.mean()
            window_peak_std = window_peak_data.std()
            window_peak_min = window_peak_data.min()
            window_peak_max = window_peak_data.max()
        else:
            # print("window_peak_count가 비어있습니다")
            window_peak_mean = 0
            window_peak_std = 0  
            window_peak_min = 0 
            window_peak_max = 0
        window_peak_feature[channel+"_peak_count"] = window_peak_count
        window_peak_feature[channel+"_peak_mean"] = window_peak_mean
        window_peak_feature[channel+"_peak_std"] = window_peak_std
        window_peak_feature[channel+"_peak_min"] = window_peak_min
        window_peak_feature[channel+"_peak_max"] = window_peak_max
    window_feature = pd.concat([window_mean, window_std, window_min, window_max, window_peak_feature], axis=0)
    return window_feature
