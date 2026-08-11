import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
# train_data = pd.read_csv("data/processed/train_data.csv")
# validation_data = pd.read_csv("data/processed/validation_data.csv")
# test_data = pd.read_csv("data/processed/test_data.csv")
#구간 단위로 묶은후 파일
train_data = pd.read_csv("data/processed/train_peak200_feature_data.csv")
validation_data = pd.read_csv("data/processed/validation_peak200_feature_data.csv")
test_data = pd.read_csv("data/processed/test_peak200_feature_data.csv")

print(test_data.shape)
print(test_data.head())

#채널과 클래스(라벨) 분리 id는 제외시켜야함
x_train = train_data.drop(columns=['ID', 'Class'])
y_train = train_data ['Class']
x_validation = validation_data.drop(columns=['ID', 'Class'])
y_validation = validation_data['Class']
x_test = test_data.drop(columns=['ID', 'Class'])
y_test = test_data['Class']

print("x_train shape : ", x_train.shape)
print("y_train shape : ", y_train.shape)
print("x_validation shape : ", x_validation.shape)
print("y_validation shape : ", y_validation.shape)
print("x_test shape : ", x_test.shape)
print("y_test shape : ", y_test.shape)
# 라벨을 숫자로 변환
y_train = y_train.replace({"ADHD": 1, "Control": 0}).astype(int)
y_validation = y_validation.replace({"ADHD": 1, "Control": 0}).astype(int)
y_test = y_test.replace({"ADHD": 1, "Control": 0}).astype(int)

#표준화
scaler = StandardScaler()
x_trainscaled = scaler.fit_transform(x_train)
x_validationscaled = scaler.transform(x_validation)
x_testscaled = scaler.transform(x_test)
# 모델 선정
# model = SVC( kernel="rbf", random_state=42)
# model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model = LogisticRegression(max_iter=3000)
# 모델 학습
model.fit(x_trainscaled, y_train)
joblib.dump(model, "model/logistic_regression_rpeak200.pk1")
joblib.dump(scaler, "model/scaler_peak200.pkl")
# 모델 평가
y_pred = model.predict(x_testscaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("test 정확도: ", accuracy)
print("test 정밀도: ", precision)
print("test 재현율: ", recall)
print("test F1 Score: ", f1)

print(y_pred[:20])
print(y_test[:20].values)
print(confusion_matrix(y_test, y_pred))

# 결과저장
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_test, y_pred)

result_df = pd.DataFrame([
    {
        "experiment": "LogisticRegression_peak200_test",
        "model": "LogisticRegression",
        "input_unit": "128_row_window_peak200",
        "feature_count": x_train.shape[1],
        "train_row_count": x_train.shape[0],
        "test_row_countt_row_count": y_test.shape[0],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "note": "실사용 데이터 하나를 제외하고 다시 prominence200 LogisticRegression test "
    }
])

result_path = results_dir / "model_results.csv"

result_df.to_csv(
    result_path,
    mode="a",
    header=not result_path.exists(),
    index=False
)