import pandas as pd
import random
# random.seed(42)

path_neighbor = "nt_nearest_neighbors.csv"
df = pd.read_csv(path_neighbor)

all_rows = []
cnt = 0
TOTAL_LEAF_NODE = 64
TOTOAL_SAMPLE = 1024

def clean_text(text):
    prefix = "Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A')."
    prefix_len = len(prefix)
    text_without_prefix = text[prefix_len:].strip()
    if "A:" in text_without_prefix:
        idx = text_without_prefix.find("A:")
        return text_without_prefix[:idx]
    return text_without_prefix

for index, row in df.iterrows():
    all_rows.append((cnt, row['Question'], row['Result']))
    if (index+1) % TOTAL_LEAF_NODE == 0:
        cnt += 1

idx = random.randint(0, TOTOAL_SAMPLE) # 795, PHY
# idx = 795

results = []
cnt = 0
for i in range(len(all_rows)):
    if idx == all_rows[i][0]:
        exercise = clean_text(all_rows[i][2])
        results.append(exercise)

data = {
    'Result': results
}

df1 = pd.DataFrame(data, columns=data.keys())
path = f'NT_Random_{idx}.xlsx'
df1.to_excel(path, index=False)
print(f"Dataframe saved at {path}.")

# df1 = pd.DataFrame(data, columns=data.keys())
# path = f'NT_Random_{idx}.csv'
# df1.to_csv(path, index=False)
# print(f"Dataframe saved at {path}.")