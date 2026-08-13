import pandas as pd 
from utils import calculate_code_hash, calculate_file_hash

def verify_file_hash(file_path, saved_hash):
    current_hash = calculate_file_hash(file_path)

    return current_hash == saved_hash