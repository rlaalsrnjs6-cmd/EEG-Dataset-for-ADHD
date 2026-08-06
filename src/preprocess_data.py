import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

raw_df = pd.read_csv("data/raw/adhd_data.csv");
print (raw_df.head())
print (raw_df.tail())
print ("원본 데이터 행 개수",raw_df.shape[0])
print("중복개수 : ",raw_df.duplicated().sum())
#중복제거
drop_data = raw_df.drop_duplicates()
print ("드랍 데이터 행 개수",drop_data.shape[0])

#결측치 제거
null_data = drop_data.dropna()
print ("결측치 제거 후 데이터 행 개수",null_data.shape[0])

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
window_size = 128
# 중복 제거 후 EEG 채널이 모두 0인 행 확인
eeg_zero = null_data[eeg_channels].eq(0).all(axis=1)
print ("EEG 채널이 모두 0인 행 개수: ", eeg_zero.sum())
# EEG 채널이 모두 0인 행 제거
# 마지막 제거 과정
eeg_drop = null_data[~eeg_zero]
print ("중복 이상치 모두 제거 후 :  ", eeg_drop.shape[0])

#특징저장용 리스트
feature_list = []

for current_id in eeg_drop['ID'].unique():
    current_id_data = eeg_drop[eeg_drop['ID'] == current_id]
#128행씩 자르기
    for start in range(0, len(current_id_data), window_size):
        window = current_id_data.iloc[start:start + window_size]
        #128행이 안되면 건너띄기
        if len(window) < window_size:
            continue
        # print(window.shape)
        window_eeg = window[eeg_channels]   
        # print(window_eeg.shape)
        #채널 특징 추출
        window_mean = window_eeg.mean()
        # print(window_mean)
        window_std = window_eeg.std()
        # print(window_std)
        window_min = window_eeg.min()
        # print(window_min)
        window_max = window_eeg.max()
        # print(window_max)1
        window_mean.index = window_mean.index + "_mean"
        window_std.index = window_std.index + "_std"
        window_min.index = window_min.index + "_min"
        window_max.index = window_max.index + "_max"
        window_feature = pd.concat([window_mean, window_std, window_min, window_max], axis=0)
        window_feature["ID"] = current_id
        window_feature["Class"] = window['Class'].iloc[0]
        feature_list.append(window_feature)

print(len(feature_list))
feature_df = pd.DataFrame(feature_list)
print(feature_df.shape)

print("데이터 정제 완료")
#id와 class 중복제거
id_class_df = feature_df[['ID', 'Class']].drop_duplicates()
print(id_class_df)
#ADHD 그룹만 추출
adhd_group = id_class_df[id_class_df['Class'] == "ADHD"]
print ("ADHD 그룹 개수")
print (adhd_group.shape[0])
#control 그룹만 추출
control_group = id_class_df[id_class_df['Class'] == "Control"]
print ("Control 그룹 개수")
print (control_group.shape[0])

#train분리
adhd_train, adhd_temp = train_test_split(adhd_group, test_size=0.3, random_state=42)
control_train, control_temp = train_test_split(control_group, test_size=0.3, random_state=42)
#validation test 분리
adhd_validation, adhd_test = train_test_split(adhd_temp, test_size=0.5, random_state=42)
control_validation, control_test = train_test_split(control_temp, test_size=0.5, random_state=42)
print("train 개수 : ", len(adhd_train) + len(control_train))
print("Validation 개수 : ", len(adhd_validation) + len(control_validation))
print("Test 개수 : ", len(adhd_test) + len(control_test))
print("================================")
#adhd그룹과 control그룹을 합쳐서 train, validation, test id를 만듬
train_id = pd.concat([adhd_train, control_train])
validation_id = pd.concat([adhd_validation, control_validation])
test_id = pd.concat([adhd_test, control_test])
print("train 개수 : ", train_id.shape[0])
print("validation 개수 : ", validation_id.shape[0])
print("test 개수 : ", test_id.shape[0])
#id토대로 데이터 가져오기
train_data = feature_df[feature_df['ID'].isin(train_id['ID'])]
validation_data = feature_df[feature_df['ID'].isin(validation_id['ID'])]
test_data = feature_df[feature_df['ID'].isin(test_id['ID'])]
# 데이터를 csv 파일로 저장
train_data.to_csv("data/processed/train_feature_data.csv", index=False)
validation_data.to_csv("data/processed/validation_feature_data.csv", index=False)
test_data.to_csv("data/processed/test_feature_data.csv", index=False)
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)




