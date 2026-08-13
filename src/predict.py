import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from data_validation import validate_eeg
from feature_extraction import extract_features
from utils import calculate_file_hash, save_log, calculate_code_hash

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
real_test_data = "data/processed/real_test_data.csv"

new_data = pd.read_csv(real_test_data)

# 에러 작동 테스트시 주석해체 그리고 하단 code_hash 전달인자 invalid_data_path로 변경 
# validation_eeg new_data를 invalid_data로 변경

# invalid_data = new_data.copy()
# invalid_data.loc[0,eeg_channels] = 0
# invalid_data.to_csv("data/processed/invalid_test_data.csv",index=False)
# invalid_data_path="data/processed/invalid_test_data.csv"
input_hash  = calculate_file_hash(new_data)
file_paths = [
    "src/predict.py",
    "src/data_validation.py",
    "src/feature_extraction.py",
    "src/utils.py"
]
code_hash = calculate_code_hash(real_test_data)
print(input_hash)
print(code_hash)
predict_log = pd.read_csv("logs/predict.csv")
latest_predict = predict_log.iloc[-1]

saved_hash = latest_predict["code_hash"]
code_hash = calculate_code_hash(file_paths)

result = code_hash == saved_hash

if result:
    print("해시값이 그대로입니다.")
else:
    print("해시값이 변했습니다.")
    
window_size = 128
prominence = 200

print(new_data["Class"].iloc[0])

validate_eeg(new_data, eeg_channels, input_hash, code_hash)

feature_list = []
for start in range(0, len(new_data), window_size):
    window = new_data.iloc[start:start + window_size]
    if len(window) < window_size:
        continue
    window_feature = extract_features(window, eeg_channels, prominence)
    feature_list.append(window_feature)
feature_df = pd.DataFrame(feature_list)

#모델 표준화객체 불러오기
model_path = "model/logistic_regression_rpeak200.pkl"
scaler_path = "model/scaler_peak200.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_hash = calculate_file_hash(model_path)
scaler_hash = calculate_file_hash(scaler_path)
print(feature_df.shape)
scaled_feature = scaler.transform(feature_df)
prediction = model.predict_proba(scaled_feature)
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
print(f"모델이 예측한 ADHD 클래스 확률의 평균 : {mean_adhd_prediction:.2%}")

predict_record ={
    "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
    "code_hash": code_hash,
    "input_hash": input_hash,
    "model_hash": model_hash,
    "scaler_hash": scaler_hash,
    "window_count": len(feature_df),
    "mean_adhd_probability": mean_adhd_prediction,

}
log_path = "logs/predict.csv"
save_log(predict_record, log_path)