import pandas as pd
import json

def load_xlsx(path):
    df = pd.read_excel(path)
    return df

def load_csv(path):
    df = pd.read_csv(path)
    return df

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_to_json(path, data):
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"File saved at {path}")