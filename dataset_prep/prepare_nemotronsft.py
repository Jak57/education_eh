import pandas as pd
from datasets import load_dataset

SCORES_CSV = "dataset_UF_with_scores.csv"
df = pd.read_csv(SCORES_CSV)

# ------------------------------------------------------------------
# Load the Nemotron-SFT-Science-v2 "so" split
# ------------------------------------------------------------------
ds = load_dataset(
    "nvidia/Nemotron-SFT-Science-v2",
    'vendor',
    split="train",
    streaming=True
)

ds = ds.shuffle(seed=42, buffer_size=65_000)
rows = [row for _, row in zip(range(len(df)), ds)]

instructions = []
subjects = []
for idx, row in enumerate(rows):
    subjects.append(row['metadata']['topic'])
    instructions.append(row['messages'][0]['content'])

df["instructions"] = instructions
df['subject'] = subjects

# ------------------------------------------------------------------
# Write the enriched CSV
# ------------------------------------------------------------------
output_file = "dataset_NT_with_scores.csv"
df.to_csv(output_file, index=False)
print(f"DataFrame successfully saved to {output_file}")