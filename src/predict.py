import pandas as pd
import joblib
from data_validation import validate_eeg
from feature_extraction import extract_features

eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
new_data = pd.read_csv("data/processed/real_test_data.csv")
# invalid_data = new_data.copy()
# invalid_data.loc[0,eeg_channels] = 0
# invalid_data.to_csv("data/proc1essed/invalid_test_data.csv",index=False)

window_size = 128
print(new_data["Class"].iloc[0])
validate_eeg(new_data, eeg_channels)
feature_list = []
for start in range(0, len(new_data), window_size):
    window = new_data.iloc[start:start + window_size]
    if len(window) < window_size:
        continue
    window_feature = extract_features(window, eeg_channels)
    feature_list.append(window_feature)
feature_df = pd.DataFrame(feature_list)

model = joblib.load("model/logistic_regression_rpeak200.pk1")
scaler = joblib.load("model/scaler_peak200.pkl")
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