import pandas as pd
import random
# random.seed(42)

path_neighbor = "nt_nearest_neighbors.csv"
df = pd.read_csv(path_neighbor)

all_rows = []
cnt = 0
TOTAL_LEAF_NODE = 64
TOTOAL_SAMPLE = 1024

for index, row in df.iterrows():
    all_rows.append((cnt, row['Question'], row['Result']))
    if (index+1) % TOTAL_LEAF_NODE == 0:
        cnt += 1

idx = random.randint(0, TOTOAL_SAMPLE) # 795, PHY
results = []
for i in range(len(all_rows)):
    if idx == all_rows[i][0]:
        results.append(all_rows[i][2])

data = {
    'Result': results
}

df1 = pd.DataFrame(data, columns=data.keys())
path = f'NT_Random_{idx}.xlsx'
df1.to_excel(path, index=False)
print(f"Dataframe saved at {path}.")