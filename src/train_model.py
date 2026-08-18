import pandas as pd
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from utils import calculate_file_hash, save_log, calculate_code_hash

#피크특징 추출 후
train_data_path = "data/processed/train_peak200_feature_data.csv"
validation_data_path = "data/processed/validation_peak200_feature_data.csv"
test_data_path = "data/processed/test_peak200_feature_data.csv"
 
train_data = pd.read_csv(train_data_path)
validation_data = pd.read_csv(validation_data_path)
test_data = pd.read_csv(test_data_path)

train_data_hash = calculate_file_hash(train_data_path)
validation_data_hash = calculate_file_hash(validation_data_path)
test_data_hash = calculate_file_hash(test_data_path)

#validation으로 할지 test로 할지
eval_type = "test"

if eval_type == "validation":
    eval_data = validation_data
    eval_hash = validation_data_hash
elif eval_type == "test":
    eval_data = test_data
    eval_hash = test_data_hash
else:
    raise ValueError("eval_type은 validation 또는 test여야 합니다.")
#채널과 클래스(라벨) 분리 id는 제외시켜야함
#EEG데이터를 보고 예측을 해야하기 때문에
x_train = train_data.drop(columns=['ID', 'Class'])
y_train = train_data ['Class']
x_eval = eval_data.drop(columns=['ID', 'Class'])
y_eval = eval_data['Class']
# x_validation = validation_data.drop(columns=['ID', 'Class'])
# y_validation = validation_data['Class']
# x_test = test_data.drop(columns=['ID', 'Class'])
# y_test = test_data['Class']

print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
print(f"x_{eval_type} shape : ", x_eval.shape)
print(f"y_{eval_type} : ", y_eval.shape)
# print("x_test shape : ", x_test.shape)
# print("y_test shape : ", y_test.shape)

# 라벨을 숫자로 변환
y_train = y_train.replace({"ADHD": 1, "Control": 0}).astype(int)
y_eval = y_eval.replace({"ADHD": 1, "Control": 0}).astype(int)
# y_validation = y_validation.replace({"ADHD": 1, "Control": 0}).astype(int)
# y_test = y_test.replace({"ADHD": 1, "Control": 0}).astype(int)

#표준화
# scaler = StandardScaler()
# x_trainscaled = scaler.fit_transform(x_train)
# x_evalscaled = scaler.transform(x_eval)
# x_validationscaled = scaler.transform(x_validation)
# x_testscaled = scaler.transform(x_test)
# 모델 선정
# model = SVC( kernel="rbf", random_state=42)
# model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
#기존 표준화와 모델 선정을 한번에
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=3000))
])
# model = LogisticRegression(max_iter=3000)
# 모델 학습
model.fit(x_train, y_train)



# 모델 평가
y_pred = model.predict(x_eval)
accuracy = accuracy_score(y_eval, y_pred)
precision = precision_score(y_eval, y_pred)
recall = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)
print("eval 정확도: ", accuracy)
print("eval 정밀도: ", precision)
print("eval 재현율: ", recall)
print("eval F1 Score: ", f1)

model_data = {
    "pipeline" :model,
    "eval_type":eval_type,
    "accuracy": accuracy,
    "precision":precision,
    "recall":recall,
    "f1":f1
}
model_path ="model/logistic_regression_rpeak200.pkl"

joblib.dump(model_data, model_path)

#해시
model_hash = calculate_file_hash(model_path)

print(y_pred[:20])
print(y_eval[:20].values)

# 결과저장
file_paths = [
    "src/train_model.py",
    "src/utils.py"
]

code_hash = calculate_code_hash(file_paths)

cm = confusion_matrix(y_eval, y_pred)

result_record ={
        "run_time": datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v1",
        "experiment": f"LogisticRegression_peak200_{eval_type}",
        "model": "LogisticRegression",
        "code_hash": code_hash,
        "model_hash": model_hash,
        "train_data_hash": train_data_hash,
        "eval_type": eval_type,
        "eval_hash": eval_hash,
        "input_unit": "128_row_window_peak200",
        "feature_count": x_train.shape[1],
        "train_row_count": x_train.shape[0],
        "eval_row_count": y_eval.shape[0],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "note": f"기록을 남기는 첫번째 버전으로써 LogisticRegression 모델 실행 {eval_type} "
    }

result_path = Path("logs/model_results.csv")
#학습 평가 저장
save_log(result_record, result_path)