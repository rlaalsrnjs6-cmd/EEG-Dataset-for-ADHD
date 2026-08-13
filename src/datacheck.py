import pandas as pd

# 해당 경로의 csv파일을 읽어옴
raw_df = pd.read_csv("data/raw/adhd_data.csv");
# 제대로 불러왔나 확인
print("raw_df.head()")
print (raw_df.head())
print("raw_df.tail()")
print (raw_df.tail())
print("raw_df.행개수")
print(raw_df.shape[0])
print("raw_df.열개수")
print(raw_df.shape[1])
# 각 id의 개수 확인
id_count = raw_df['ID'].nunique()
print ("id 개수 : ", id_count)
# 각 id별 클래스 개수 확인(id가 class를 두개 가지면 잘못된 데이터)
class_count = raw_df.groupby('ID')["Class"].nunique()
# 클래스가 1개 이상인 경우가 존재하는지 확인
for id, count in class_count.items():
    if count > 1:
        print("id별 클래스 개수가 1개이상인 경우가 존재 합니다")
print ("중복확인")
# 중복 데이터 개수 확인
print (raw_df.duplicated().sum())
# 중복 데이터가 뭔지 확인
# print (raw_df[raw_df.duplicated()])
print(raw_df[raw_df.duplicated(keep=False)])
print("결측치 개수")
print(raw_df.isnull().sum())
# 평균 표균편차 최소 최대 값 확인
print("평균 표준편차 최소 최대 값 확인")
print(raw_df.describe())

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
# 한 행의 열이 모두 0이라면 이상치라 생각 해서 제외
eeg_zero = raw_df[eeg_channels].eq(0).all(axis=1)
print ("EEG 채널이 모두 0인 행 개수: ", eeg_zero.sum())
print ("EEG 채널이 모두 0인 행 확인")
print (raw_df[eeg_zero])