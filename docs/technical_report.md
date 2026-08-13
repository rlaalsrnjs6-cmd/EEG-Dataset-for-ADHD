# 1. 문서 목적

- 본 문서는 EEG 데이터를 활용하여 ADHD와 Control을 분휴하기 위해 수행한 데이터 전처리, 특징 추출, 머신러닝 모델 비교, 성능 평가 및 최종 모델 선정 과정을 정리한다.

# 2. 모델의 목적

- EEG데이터를 통한 ADHD, Control 예측한다.

# 3. 모델 입력 데이터

- 원본 EEG데이터를 ID로 구분한 후 128행 단위의 Window로 분할한다.
- 각 Window의 EEG 신호를 모델에 직접 입력하지 않고 각 EEG채널에서 통계 특징과 Peak 특징을 추출하여 머신러닝 모델의 입력 데이터로 사용한다.
- 19개의 EEG 채널에서 채널당 9개의 특징을 추출하므로 하나의 Window에는 총 171개의 Feature로 반환된다.
- ID는 피험자를 구분하고 Train, Validation, Test 데이터를 피험자 단위로 분리하기 위해 사용하며 머신러닝모델의 입력 Feature에서는 제외된다.

## 3.1 통계 특징

- 평균(Mean)
- 표준편차(Standard Deviation)
- 최솟값(Min)
- 최댓값(Max)

## 3.2 피크 특징
- 
- Peak 개수(Count)
- Peak 평균(Mean)
- Peak 표준편차(Standard Deviation)
- Peak 최솟값(Min)
- Peak 최댓값(Max)

# 4. 비교 머신러닝 모델

- Logistic Regression
- Random Forest
- SVM

# 5. 모델 설정값

- random_state
- n_estimators
- class_weight
- kernel
- max_iter

# 6. 학습 데이터 구성

- 원본의 128Hz의 데이터를 128행의 단위로 묶어 특징 추출 (원본 EEG데이터의 SamplingRate가 128Hz이므로 128행을 하나의 Window로 구성하여 약 1초 단위의 신호 구간으로 특징 추출)
- Train / Validation / Test 구분
- 같은 ID가 다른 데이터셋에 섞이지않게 ID를 기준으로 구분 (같은 피험자의 EEG데이터가 Train/Validation/Test에 동시에 포하묄 경우 데이터 누수가 발생할 수 있으므로)
- 70% / 15% / 15% 비율로 구분

# 7. 학습 및 평가 과정

- Train으로 학습
- Validation으로 비교
- Test로 최종 평가

# 8. 평가 지표

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

# 9. 실험 결과

- 피크 특징 적용전 실험결과

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | EEG의 한행을 그대로 | 19 | 0.6015 | 0.6016 | 0.9999 | 0.7511 |
| LogisticRegression | EEG의 한행을 그대로 표준화 진행 | 19 | 0.6015 | 0.6016 | 0.9996 | 0.7511 |
| LogisticRegression | 128행단위로 구간을 묶고 평균과 표준편차 특징으로 학습 StandardScaler 적용x | 38 | 0.6886 | 0.7193 | 0.7914 | 0.7536 |
| LogisticRegression | 128행단위로 구간을 묶고 평균과 표준편차 최솟값 최대값 특징으로 학습 StandardScaler 적용x | 76 | 0.6900 | 0.7254 | 0.7803 | 0.7519 |
| LogisticRegression | 128행단위로 구간을 묶고 평균과 표준편차 최솟값 최대값 특징으로 학습 StandardScaler 적용 | 76 | 0.6893 | 0.7242 | 0.7814 | 0.7517 |
| RandomForest | RandomForest모델변경 | 76 | 0.6354 | 0.6917 | 0.7111 | 0.7013 |
| SVC | SVC모델변경 | 76 | 0.6791 | 0.7294 | 0.7422 | 0.7357 |

- 피크 특징 적용 후 실험결과

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | Prominence500 피크 평균 표준편차 최솟값 최대값 피크개수 특징 추출후 결합 | 171 | 0.6960 | 0.7124 | 0.8301 | 0.7667 |
| LogisticRegression | Prominence300 | 171 | 0.7644 | 0.7715 | 0.8646 | 0.8154 |
| LogisticRegression | Prominence1000 | 171 | 0.6717 | 0.7172 | 0.7504 | 0.7334 |
| LogisticRegression | Prominence200 | 171 | 0.8169 | 0.8296 | 0.8758 | 0.8520 |
| LogisticRegression | Prominence100 | 171 | 0.7559 | 0.8205 | 0.7609 | 0.7896 |
| LogisticRegression | Prominence150 | 171 | 0.7789 | 0.8264 | 0.8008 | 0.8134 |
| LogisticRegression | Prominence250 | 171 | 0.7916 | 0.7996 | 0.8722 | 0.8344 |
| RandomForest | Prominence200 | 171 | 0.7150 | 0.7693 | 0.7521 | 0.7606 |
| SVC | Prominence200 | 171 | 0.7147 | 0.7647 | 0.7598 | 0.7622 |

- 실사용 ID 제거후 재분류 한 후

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | Prominence200 | 171 | 0.7374 | 0.7579 | 0.7449 | 0.7513 |

- 마지막 Test

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | Prominence200후 test | 171 | 0.7428 | 0.7301 | 0.8597 | 0.7896 |

# 10. 최종 모델 선정

- LogisticRegression
- prominence=200 171 Feature의 조건에서 LogisticRegression이 가장 높은 성능을 보였다.

# 11. 모델 저장 및 사용

- 학습된 모델을 logistic_regression_rpeak200.pk1 저장
- Scaler 기준 scaler_peak200.pkl 저장
- predict.py에서 모델과 Scaler를 불러와 실사용 데이터를 표준화를 거쳐 입력

# 12. 모델의 한계

- 실사용 ID를 한 명 제외하고 다시 분할을 했을때 성능이 달라졌다.
- 어떤 피험자가 분리되는지에 따라 성능이 영향을 받는다.
- 시간 순서를 직접확인 하지않고 평균 표준편차 peak등의 특징으로 변환해야한다.
- 현재 데이터로 prominence=200이 가장 좋은 결과를 냈지만 다른 데이터에서 200이 최적이라 보장 못함

# 13. 재현 정보

- python 3.14.6
- pandas, scikit-learn, matplotlib.pyplot, numpy, joblib
- random_state = 42
- train/validation/test feature CSV
- model/logistic_regression_rpeak200.pk1
- model/scaler_peak200.pkl
- datacheck.py(검증) -> peak_detection.py(피크 확인) -> preprocess_data.py(전처리 및 분류) -> train_model.py(학습 및 평가) -> predict.py(데이터 실사용 에러 발생상황 재현시 new_data경로의 real_test_data.csv를 invalid_test_data.csv로 변경)