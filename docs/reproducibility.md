# 1. 재현 정보

- python 3.14.6
- 데이터 분할 random_state = 42
- train/validation/test feature CSV
- model/logistic_regression_rpeak200.pkl
- model/scaler_peak200.pkl

# 2. 실행 순서

datacheck.py(데이터 검증) 
-> peak_detection.py(Peak 확인) 
-> preprocess_data.py(전처리 및 데이터 분할) 
-> train_model.py(학습 및 평가) 
-> predict.py(저장된 모델을 통해 실사용 데이터 예측)

# 3. 에러 상황 재현

- input_hash의 전달인자를 `invalid_data_path`로 변경
- validation_eeg의 전달인자를 `new_data`에서 `invalid_data`로 변경

# 4. 해시 검증

- verify_hash.py 실행

# 4.1 해시 검증 정보

- 해시 알고리즘: SHA-256
- 입력 데이터와 실행 코드의 해시값을 생성하여 변경 여부를 확인한다.
- 생성된 해시값은 logs 폴더의 로그 파일에 기록한다.

# 5. 사용 라이브러리

- pandas: 3.0.5
- scipy: 1.18.0
- matplotlib: 3.11.1
- joblib: 1.5.3
- scikit-learn: 1.9.0
- datetime: Python 표준 라이브러리
- zoneinfo: Python 표준 라이브러리
- pathlib: Python 표준 라이브러리
- hashlib: Python 표준 라이브러리