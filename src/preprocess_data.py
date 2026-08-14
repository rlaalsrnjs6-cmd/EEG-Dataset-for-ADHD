from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features
from utils import calculate_file_hash, save_log, calculate_code_hash

#데이터 읽어오기
raw_data_path = "data/raw/adhd_data.csv"
raw_df = pd.read_csv(raw_data_path)
#해시계산
input_hash = calculate_file_hash(raw_data_path)
print("원본 데이터 해시 :", input_hash)
print ("원본 데이터 행 개수",raw_df.shape[0])
print("중복개수 : ",raw_df.duplicated().sum())
#중복제거
drop_data = raw_df.drop_duplicates()
print ("중복제거 후 데이터 행 개수 : ",drop_data.shape[0])

#결측치 제거
null_data = drop_data.dropna()
print ("결측치 제거 후 데이터 행 개수 : ",null_data.shape[0])

duplicate_count = len(raw_df) - len(drop_data)
null_count = len(drop_data) - len(null_data)
#128행이 안되서 버려지는 행의 수
incomplete_window_row_count = 0

#원본 데이터 채널 컬럼 이름
eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
#데이터가 128hz단위로 체크했기에 128행씩 데이터를 합쳐서 학습시키기위함
#id가 가진 고유 패턴을 학습하기 위해서

window_size = 128
random_state = 42
prominence = 200

# 중복 제거 후 EEG 채널이 모두 0인 행 확인
eeg_zero = null_data[eeg_channels].eq(0).all(axis=1)
eeg_zero_count = eeg_zero.sum()
print("eeg_zero : ", eeg_zero_count)
# EEG 채널이 모두 0인 행 제거
# 마지막 제거 과정
eeg_drop = null_data[~eeg_zero]
print ("중복 이상치 모두 제거 후 :  ", eeg_drop.shape[0])
#실사용 데이터
real_test_id = eeg_drop['ID'].unique()[0]
real_test_data = eeg_drop[eeg_drop['ID'] == real_test_id]
#실사용 데이터를 제외한 데이터
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
            incomplete_window_row_count += len(window)
            continue
        # print(window.shape)
        #함수로 교체 특징(EEG데이터 특징, 피크 특징) 추출
        window_feature = extract_features(window, eeg_channels, prominence)
        #이값들의 id와 class도 추가
        window_feature["ID"] = current_id
        window_feature["Class"] = window['Class'].iloc[0]
        #리스트담기
        feature_list.append(window_feature)


print(len(feature_list))
#시리즈를 df형식으로 변환
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
adhd_train, adhd_temp = train_test_split(adhd_group, test_size=0.3, random_state=random_state)
control_train, control_temp = train_test_split(control_group, test_size=0.3, random_state=random_state)

#validation test 분리
adhd_validation, adhd_test = train_test_split(adhd_temp, test_size=0.5, random_state=random_state)
control_validation, control_test = train_test_split(control_temp, test_size=0.5, random_state=random_state)

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
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape) 

# 데이터를 csv 파일로 저장
train_data_path = "data/processed/train_peak200_feature_data.csv"
validation_data_path = "data/processed/validation_peak200_feature_data.csv"
test_data_path = "data/processed/test_peak200_feature_data.csv"
real_test_data_path = "resource/real_test_data.csv"

train_data.to_csv(train_data_path, index=False)
validation_data.to_csv(validation_data_path, index=False)
test_data.to_csv(test_data_path, index=False)
real_test_data.to_csv(real_test_data_path, index=False)

train_hash = calculate_file_hash(train_data_path)
validation_hash = calculate_file_hash(validation_data_path)
test_hash = calculate_file_hash(test_data_path)
real_test_hash = calculate_file_hash(real_test_data_path)

#코드 버전 관리 참조 파일 저장
file_paths = [
    "src/preprocess_data.py",
    "src/feature_extraction.py",
    "src/utils.py"
    ]

code_hash = calculate_code_hash(file_paths)

# 전처리 딕셔너리
preprocess_record = {
    "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
    "input_hash": input_hash,

    "window_size":window_size,
    "prominence":prominence,
    "random_state": random_state,

    "duplicate_count": duplicate_count,
    "null_count": null_count,
    "eeg_zero_count": eeg_zero_count,
    "incomplete_window_row_count": incomplete_window_row_count,

    "code_hash": code_hash,
    "train_hash": train_hash,
    "validation_hash": validation_hash,
    "test_hash": test_hash,
    "real_test_hash": real_test_hash
}
#파일위치
log_path = Path("logs/preprocess_log.csv")
#전처리 결과물 저장
save_log(preprocess_record, log_path)

