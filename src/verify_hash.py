import pandas as pd 
from utils import calculate_code_hash, calculate_file_hash

def verify_file_hash(file_path, saved_hash):
    current_hash = calculate_file_hash(file_path)

    return current_hash == saved_hash

def verify_code_hash(file_paths, saved_hash):
    current_hash = calculate_code_hash(file_paths)

    return current_hash == saved_hash

file_paths = [
    "src/preprocess_data.py",
    "src/feature_extraction.py",
    "src/utils.py"
]

#preprocess_log
raw_data_path = "data/raw/adhd_data.csv"
train_data_path = "data/processed/train_peak200_feature_data.csv"
validation_data_path = "data/processed/validation_peak200_feature_data.csv"
test_data_path = "data/processed/test_peak200_feature_data.csv"
real_test_data_path = "data/processed/real_test_data.csv"

preprocess_log = pd.read_csv("logs/preprocess_log.csv")
latest_preprocess = preprocess_log.iloc[-1]

#preprocess_log
raw_result = verify_file_hash(raw_data_path, latest_preprocess['input_hash'])
train_result = verify_file_hash(train_data_path, latest_preprocess['train_hash'])
validation_result = verify_file_hash(validation_data_path, latest_preprocess['validation_hash'])
test_result = verify_file_hash(test_data_path, latest_preprocess['test_hash'])
real_test_result = verify_file_hash(real_test_data_path, latest_preprocess['real_test_hash'])
preprocess_code_result = verify_code_hash(file_paths, latest_preprocess["code_hash"])

#train_model
train_code_paths = [
    "src/train_model.py",
    "src/utils.py"
]

model_path = "model/logistic_regression_rpeak200.pkl"
scaler_path = "model/scaler_peak200.pkl"

model_log = pd.read_csv("logs/model_results.csv")
latest_model = model_log.iloc[-1]

model_result = verify_file_hash(model_path, latest_model["model_hash"])
scaler_result = verify_file_hash(scaler_path, latest_model["scaler_hash"])
model_code_result = verify_code_hash(train_code_paths, latest_model["code_hash"])

#predict
predict_file_paths = [
    "src/predict.py",
    "src/data_validation.py",
    "src/feature_extraction.py",
    "src/utils.py"
]
predict_log = pd.read_csv("logs/predict.csv")
latest_predict = predict_log.iloc[-1]
predict_result = verify_code_hash(predict_file_paths, latest_predict["code_hash"])

print("원본 데이터 :", "정상" if raw_result else "변경됨")
print("Train 데이터 :", "정상" if train_result else "변경됨")
print("Validation 데이터 :", "정상" if validation_result else "변경됨")
print("Test 데이터 :", "정상" if test_result else "변경됨")
print("Real Test 데이터 :", "정상" if real_test_result else "변경됨")

print("Model :", "정상" if model_result else "변경됨")
print("Scaler :", "정상" if scaler_result else "변경됨")

print("Preprocess 코드 :", "정상" if preprocess_code_result else "변경됨")
print("Train 코드 :", "정상" if model_code_result else "변경됨")
print("Predict 코드 :", "정상" if predict_result else "변경됨")