import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

raw_df = pd.read_csv("data/raw/adhd_data.csv")

random_state = 42
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

window_list = []
label_list = []
id_list = []

for current_id in eeg_data["ID"].unique():
    id_data = eeg_data[eeg_data["ID"]==current_id]
    for start in range(0, len(id_data), window_size):
        window_data = id_data.iloc[start: start+window_size]
        if len(window_data) < window_size:
            continue
        # 전치후 넘파이 배열로 변경
        window_eeg = window_data[eeg_channels].T.to_numpy()
        window_label = window_data["Class"].iloc[0]
        window_id = window_data["ID"].iloc[0]
        window_list.append(window_eeg)
        label_list.append(window_label)
        id_list.append(window_id)

label_list = [ 
    1 if label == "ADHD" else 0
    for label in label_list
]

print(len(window_list))
print(len(label_list))
print(len(id_list))

id_class_df = pd.DataFrame({
    "ID": id_list,
    "Class": label_list
}).drop_duplicates()

real_data = id_class_df[id_class_df["Class"] == 1]["ID"].iloc[0]
adhd_grops = id_class_df[(id_class_df["Class"] == 1) & (id_class_df["ID"] != real_data)]["ID"]
control_grops = id_class_df[id_class_df["Class"] == 0]["ID"]
print(len(adhd_grops))
print(len(control_grops))

adhd_train, adhd_temp = train_test_split(adhd_grops, test_size=0.3, random_state=random_state)
adhd_validation, adhd_test = train_test_split(adhd_temp, test_size=0.5, random_state=random_state)
control_train, control_temp = train_test_split(control_grops, test_size=0.3, random_state=random_state)
control_validation, control_test = train_test_split(control_temp, test_size=0.5, random_state=random_state)

train_ids = pd.concat([adhd_train, control_train])
validation_ids = pd.concat([adhd_validation, control_validation])
test_ids = pd.concat([adhd_test, control_test])
print(len(train_ids))
print(len(validation_ids))
print(len(test_ids))

train_windows = []
train_labels = []

validation_windows = []
validation_labels = []

test_windows = []
test_labels = []

real_test_windows = []
real_test_labels = []

for window, label, current_id in zip(window_list, label_list, id_list):
    if current_id in train_ids.values:
        train_windows.append(window)
        train_labels.append(label)
    if current_id in validation_ids.values:
        validation_windows.append(window)
        validation_labels.append(label)
    if current_id in test_ids.values:
        test_windows.append(window)
        test_labels.append(label)
    if current_id == real_data:
        real_test_windows.append(window)
        real_test_labels.append(label)

print(len(train_windows), len(train_labels))
print(len(validation_windows), len(validation_labels))
print(len(test_windows), len(test_labels))
print(len(real_test_windows), len(real_test_labels))

train_array = np.stack(train_windows)
validation_array = np.stack(validation_windows)
test_array = np.stack(test_windows)
real_array = np.stack(real_test_windows)

np.savez(
    "data/cnn_processed/cnn_data.npz",
    train_windows=train_array,
    train_labels=np.array(train_labels),
    validation_windows=validation_array,
    validation_labels=np.array(validation_labels),
    test_windows=test_array,
    test_labels=np.array(test_labels),
    real_windows=real_array,
    real_labels=np.array(real_test_labels)
)

