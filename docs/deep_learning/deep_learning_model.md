# 1. 모델 목적

- EEG 데이터를 통한 ADHD, Control을 예측

# 2. 모델 입력 데이터

- Sampling Rate 128Hz.
- ID별로 데이터 구분.
- 128행 Window 생성.
- EEG 채널 19개 사용.
- 하나의 Window 입력 구조 (19, 128).

# 3. 사용 딥러닝 모델 

- PyTorch 기반 1D CNN

# 4. 데이터 구성

- Train / Validation / Test / Real
- NumPy -> PyTorch Tensor 변환
- TensorDataset, DataLoader 사용

# 7. 평가 지표

- Loss
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix