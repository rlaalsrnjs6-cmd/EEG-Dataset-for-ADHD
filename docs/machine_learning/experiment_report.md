# 1. 실험 결과

- 구간단위 분할 적용전 실험결과

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | EEG의 한행을 그대로 | 19 | 0.6015 | 0.6016 | 0.9999 | 0.7511 |
| LogisticRegression | EEG의 한행을 그대로 표준화 진행 | 19 | 0.6015 | 0.6016 | 0.9996 | 0.7511 |

- 구간 단위 Window 설정 후 통계특징 추출 실험 결과

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
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

- XG부스터 모델 변경

| 모델 | 단계 | Feature | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost |  XGBoost 모델 변경후 validation 테스트 | 171 | 0.6190 | 0.6208 | 0.7309 | 0.6714 |

# 2. 결론

- 1행 단위의 EEG 데이터를 그대로 학습했을 경우 대부분의 예측값을 ADHD로 결정하는 문제가 있었다.
- 128행 Window단위로 통계특징을 추출하자 위 문제가 해결되었다.
- Peak 특징을 추출하자 성능 향상을 확인했다.
- Peak의 prominence가 200일때 가장 높은 성능을 보였다.
- 실사용 ID를 제거하고 분류 및 학습을 하자 성능이 떨어짐을 보였다.
- 데이터 분류에 따른 성능의 오차가 있는것으로 보인다.