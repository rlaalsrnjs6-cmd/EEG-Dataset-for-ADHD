import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from data_validation import validate_eeg
from feature_extraction import extract_features
from utils import calculate_file_hash, save_log, calculate_code_hash

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
real_test_data = "resource/real_test_data.csv"

new_data = pd.read_csv(real_test_data)

# 에러 작동 테스트시 주석해체 그리고 하단 input_hash는 invalid_data_path로 바꿔주기 
# validation_eeg new_data를 invalid_data로 변경
invalid_data = new_data.copy()
invalid_data.loc[0,eeg_channels] = 0
invalid_data.to_csv("data/processed/invalid_test_data.csv",index=False)
invalid_data_path="data/processed/invalid_test_data.csv"

# 실사용 데이터의 해시
input_hash  = calculate_file_hash(real_test_data)
# 코드 해시를 만들기위한 파일
file_paths = [
    "src/predict.py",
    "src/data_validation.py",
    "src/feature_extraction.py",
    "src/utils.py"
]
# 코드해시 생성
code_hash = calculate_code_hash(file_paths)
print(input_hash)
print(code_hash)


window_size = 128
prominence = 200

print(new_data["Class"].iloc[0])

validate_eeg(new_data, eeg_channels, input_hash, code_hash)

feature_list = []
# 128행 나누기
for start in range(0, len(new_data), window_size):
    window = new_data.iloc[start:start + window_size]
    if len(window) < window_size:
        continue
    window_feature = extract_features(window, eeg_channels, prominence)
    feature_list.append(window_feature)
feature_df = pd.DataFrame(feature_list)

#모델 표준화객체 불러오기
model_path = "model/logistic_regression_rpeak200.pkl"
# scaler_path = "model/scaler_peak200.pkl"
model_data = joblib.load(model_path)
model = model_data["pipeline"]
accuracy = model_data["accuracy"]
eval_type = model_data["eval_type"]
# scaler = joblib.load(scaler_path)
model_hash = calculate_file_hash(model_path)
# scaler_hash = calculate_file_hash(scaler_path)
print(feature_df.shape)
# 표준화된 특징
# scaled_feature = scaler.transform(feature_df)
# 모델에 입력 후 값
prediction = model.predict_proba(feature_df)
# prediction = model.predict(scaled_feature)

# adhd_count = (prediction == 1).sum()
# control_count = (prediction == 0).sum()

# print(prediction)
# print("ADHD 예측 window:", adhd_count)
# print("Control 예측 window:", control_count)

# print(prediction)
print(prediction.shape)
print(model.classes_)

adhd_prediction = prediction[:, 1]
mean_adhd_prediction = adhd_prediction.mean()
print("==========================================================================================")
print("※ 본 모델의 예측 결과는 판단을 보조하기 위한 참고 정보이며, ADHD 진단을 의미하지 않습니다.")
print("==========================================================================================")

print(f"모델의 {eval_type} 정확도 : {accuracy:.2%}")
print(f"모델이 예측한 ADHD 클래스 확률의 평균 : {mean_adhd_prediction:.2%}")

# 저장용 딕셔너리
predict_record ={
    "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
    "code_hash": code_hash,
    "input_hash": input_hash,
    "model_hash": model_hash,
    "window_count": len(feature_df),
    "mean_adhd_probability": mean_adhd_prediction,
    "note": "오류코드 실행 eroor_log가 작동하는지 확인"
}
log_path = "logs/predict.csv"
save_log(predict_record, log_path)