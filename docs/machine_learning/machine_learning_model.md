# 1. 모델 목적

- EEG 데이터를 통한 ADHD, Control을 예측

# 2. 모델 입력 데이터

- 원본 EEG 데이터의 Sampling Rate는 128Hz이다.
- 원본 EEG 데이터를 ID별로 구분한 뒤 128행의 Window로 구분한다.
- Window의 데이터에서 통계 및 피크 특징을 추출하여 모델의 입력 데이터로 사용한다.
- 19개의 EEG 채널에서 채널당 9개의 특징을 추출하므로 하나의 Window에서 총 171개의 Feature를 생성한다.
- ID는 피험자의 구분을 위해 사용하며 모델의 Feature입력에는 제외한다.

## 2.1 통계 특징

- 평균(Mean)
- 표준편차(Standard Deviation)
- 최솟값(Min)
- 최댓값(Max)

## 2.2 피크 특징

- Peak 개수(Count)
- Peak 평균(Mean)
- Peak 표준편차(Standard Deviation)
- Peak 최솟값(Min)
- Peak 최댓값(Max)

# 3. 비교 머신러닝 모델

- Logistic Regression
- Random Forest
- SVM

# 4. 모델 및 특징 추출 설정값

## 4.1 Logistic Regression

- max_iter = 3000

## 4.2 Random Forest

- n_estimators = 200
- random_state = 42
- class_weight = "balanced"

## 4.3 SVM

- kernel = "rbf"
- random_state=42

## 4.4 Peak 탐지

- prominence = 200

# 5. 학습 데이터 구성

- 데이터 분할의 random_state를 42로 설정하여 동일한 조건에서 분할 결과를 재현할 수 있도록 한다. 
- 데이터를 Train / Validation / Test로 구분한다.
- Class를 통해 ADHD 그룹과 Control 그룹을 구분하여 한곳에 한 그룹이 편중되는 것을 방지한다.
- ADHD / Control 그룹을 각각 Train / Validation / Test를 70% / 15% / 15% 비율로 구성한다.
- 각각 만들어진 Train / Validation / Test를 합쳐 하나의 Train / Validation / Test를 만든다. 
- ID를 통해 데이터를 분할하여 하나의 ID 데이터가 Train / Validation / Test에 동시에 포함되어 데이터 누수가 일어 나는 것을 방지한다.

# 6. 학습 및 평가 과정

- Train으로 학습
- Validation으로 비교
- Test로 최종 평가

# 7. 평가 지표

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix