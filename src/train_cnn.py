import pandas as pd

raw_data = pd.read_csv("data/raw/adhd_data.csv")

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']

eeg_id = raw_data['ID'].iloc[0]

eeg_id_data = raw_data[raw_data['ID']==eeg_id]

window_list = []
label_list = []

window_size = 128

for eeg_data in raw_data:
    eeg_id = eeg_data[eeg_data['ID']].unique()
    eeg_id_data = eeg_data[eeg_id]
    
    for start in range(0, len(eeg_id_data), window_size):
        window_data = eeg_id_data.iloc[start:start+window_size]
        if len(window_data) < window_size:
            continue
        eeg_channels_data = window_data[eeg_channels].to_numpy()
        lavel_list = window_data['Class']
        window_list = eeg_channels_data
        lavel_list = 
        print(eeg_channels_data.shape)

