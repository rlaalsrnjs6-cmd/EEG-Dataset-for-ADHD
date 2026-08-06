import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# train_data = pd.read_csv("data/processed/train_data.csv")
# validation_data = pd.read_csv("data/processed/validation_data.csv")
# test_data = pd.read_csv("data/processed/test_data.csv")
#구간 단위로 묶은후 파일
train_data = pd.read_csv("data/processed/train_feature_data.csv")
validation_data = pd.read_csv("data/processed/validation_feature_data.csv")
test_data = pd.read_csv("data/processed/test_feature_data.csv")

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
model = SVC( kernel="rbf", random_state=42)
# model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
# model = LogisticRegression(max_iter=3000)
# 모델 학습
model.fit(x_trainscaled, y_train)
# 모델 평가
y_pred = model.predict(x_validationscaled)
accuracy = accuracy_score(y_validation, y_pred)
precision = precision_score(y_validation, y_pred)
recall = recall_score(y_validation, y_pred)
f1 = f1_score(y_validation, y_pred)
print("Validation 정확도: ", accuracy)
print("Validation 정밀도: ", precision)
print("Validation 재현율: ", recall)
print("Validation F1 Score: ", f1)

print(y_pred[:20])
print(y_validation[:20].values)
print(confusion_matrix(y_validation, y_pred))

# 결과저장
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

cm = confusion_matrix(y_validation, y_pred)

result_df = pd.DataFrame([
    {
        "experiment": "SVC_01",
        "model": "SVC",
        "input_unit": "128_row_window",
        "feature_count": x_train.shape[1],
        "train_row_count": x_train.shape[0],
        "validation_row_count": x_validation.shape[0],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "note": "SVC모델로 변경"
    }
])

result_path = results_dir / "model_results.csv"

result_df.to_csv(
    result_path,
    mode="a",
    header=not result_path.exists(),
    index=False
)