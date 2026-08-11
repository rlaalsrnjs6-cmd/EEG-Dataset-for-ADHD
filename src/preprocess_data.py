import pandas as pd
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features

#데이터 읽어오기
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
#데이터가 128hz단위로 체크했기에 128행씩 데이터를 합쳐서 학습시키기위함
window_size = 128
# 중복 제거 후 EEG 채널이 모두 0인 행 확인
eeg_zero = null_data[eeg_channels].eq(0).all(axis=1)
print ("EEG 채널이 모두 0인 행 개수: ", eeg_zero.sum())
# EEG 채널이 모두 0인 행 제거
# 마지막 제거 과정
eeg_drop = null_data[~eeg_zero]
print ("중복 이상치 모두 제거 후 :  ", eeg_drop.shape[0])

#실사용 데이터
real_test_id = eeg_drop['ID'].unique()[0]
real_test_data = eeg_drop[eeg_drop['ID'] == real_test_id]
model_data = eeg_drop[eeg_drop['ID'] != real_test_id]
#특징저장용 리스트
feature_list = []

for current_id in model_data['ID'].unique():
    #id별 데이터 가져오기
    current_id_data = model_data[model_data['ID'] == current_id]
    
    for start in range(0, len(current_id_data), window_size):
        #128행씩 자르기 window=128행의 모든 데이터
        window = current_id_data.iloc[start:start + window_size]
        #128행이 안되면 건너띄기 
        if len(window) < window_size:
            continue
        # print(window.shape)
        #함수로 교체
        window_feature = extract_features(window, eeg_channels)
        #이값들의 id와 class도 추가
        window_feature["ID"] = current_id
        window_feature["Class"] = window['Class'].iloc[0]
        #리스트담기
        feature_list.append(window_feature)

print(len(feature_list))
feature_df = pd.DataFrame(feature_list)
print("특징개수 : ", feature_df.shape)

print("데이터 정제 완료")

#id와 class 중복제거 id가 섞이지 않게 하기위함
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
#아직 id와 class만 가지고 있음
train_id = pd.concat([adhd_train, control_train])
validation_id = pd.concat([adhd_validation, control_validation])
test_id = pd.concat([adhd_test, control_test])
print("train 개수 : ", train_id.shape[0])
print("validation 개수 : ", validation_id.shape[0])
print("test 개수 : ", test_id.shape[0])
#id토대로 데이터 가져오기 feature_df는 모든 데이터를 가지고 있음 비교해서 id가 같은 데이터만 뽑기
train_data = feature_df[feature_df['ID'].isin(train_id['ID'])]
validation_data = feature_df[feature_df['ID'].isin(validation_id['ID'])]
test_data = feature_df[feature_df['ID'].isin(test_id['ID'])]
# 데이터를 csv 파일로 저장
train_data.to_csv("data/processed/train_peak200_feature_data.csv", index=False)
validation_data.to_csv("data/processed/validation_peak200_feature_data.csv", index=False)
test_data.to_csv("data/processed/test_peak200_feature_data.csv", index=False)
real_test_data.to_csv("data/processed/real_test_data.csv", index=False)
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape) 