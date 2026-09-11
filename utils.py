import pandas as pd

def load_xlsx(path):
    df = pd.read_excel(path)
    exercises = []
    for index, row in df.iterrows():
        exercises.append(row['Result'])
    return exercises