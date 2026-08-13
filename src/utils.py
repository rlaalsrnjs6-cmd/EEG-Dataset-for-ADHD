import hashlib
import pandas as pd
from pathlib import Path

#로그저장
def save_log(record, log_path):
    log_path = Path(log_path)
    #상위폴더가없다면 만들고 logs폴더가 있어도 오류를 내지마라
    log_path.parent.mkdir(parents=True, exist_ok=True)
    #딕셔너리를 df로 변경
    log_df = pd.DataFrame([record])

    log_df.to_csv(  
        log_path,
        #append
        mode="a",
        #파일이 없으면 컬럼명을 쓰고 있으면 컬럼명을 쓰지마라
        header=not log_path.exists(),
        index=False,
        encoding="utf-8-sig"
    )

#해시 계산
def calculate_file_hash(file_path):
    sha256 = hashlib.sha256()
        #파일을 바이트로 읽겠다 
    with open(file_path, "rb") as file:
        while True:
            #8192바이트 단위로 읽겠다
            chunk = file. read(8192)

            if not chunk:
                break

            sha256.update(chunk)
            #문자열 형태로 바꾸기
    return sha256.hexdigest()

#코드해시생성
def calculate_code_hash(file_paths):
    sha256 = hashlib.sha256()

    for file_path in file_paths:
        print(file_path)
        with open(file_path, "rb") as file:
            while True:
                chunk = file.read(8192)

                if not chunk:
                    break

                sha256.update(chunk)
    return sha256.hexdigest()

#해시비교
def verify_file_hash(file_path, saved_hash):
    current_hash = calculate_file_hash(file_path)

    return current_hash == saved_hash

#코드해시비교
def verify_code_hash(file_paths, saved_hash):
    current_hashs = calculate_code_hash(file_paths)
    
    return current_hashs == saved_hash