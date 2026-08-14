# 1. 모델

- LogisticRegression
- Peak prominence를 여러 값으로 비교한 결과 prominence=200에서 가장 높은 Validation 성능을 확인하였다. 이후 동일 조건에서 Random Forest와 SVM을 비교한 결과 LogisticRegression의 성능이 가장 높아 최종 모델로 선정하였다.

# 2. 모델 저장 및 사용

- 학습된 모델을 logistic_regression_rpeak200.pkl 저장
- Scaler 기준 scaler_peak200.pkl 저장
- predict.py에서 모델과 Scaler를 불러와 실사용 데이터를 표준화를 거쳐 모델에 입력

# 3. 모델의 한계

- 실사용 ID를 한 명 제외하고 다시 분할을 했을 때 성능이 달라졌다.
- 어떤 피험자가 분리되는지에 따라 성능이 영향을 받는다.
- 128행 Window의 통계 및 peak특징을 추출하여 사용하므로 시간적 순서, 신호 변화 패턴을 학습하지 못한다.
- 현재 데이터로 prominence=200이 가장 좋은 결과를 냈지만 다른 데이터에서 200이 최적이라 보장할 수 없다.

# 4. 최종 테스트 성능
- Accuracy  : 0.7428
- Precision : 0.7301
- Recall    : 0.8597
- F1-score  : 0.7896