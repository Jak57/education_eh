import pandas as pd
import json
from datasets import load_dataset
from pathlib import Path
import random
import unicodedata
import re

_ALLOWED_LETTER_PREFIXES = ("LATIN", "GREEK")

def prepare_dataset_nemotron_so():
    SCORES_CSV = "dataset_UF_with_scores.csv"
    df = pd.read_csv(SCORES_CSV)
    # ------------------------------------------------------------------
    # Load the Nemotron-SFT-Science-v2 "so" split
    # ------------------------------------------------------------------
    ds = load_dataset(
        "nvidia/Nemotron-SFT-Science-v2",
        'so',
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
    # df['subject'] = subjects
    # ------------------------------------------------------------------
    # Write the enriched CSV
    # ------------------------------------------------------------------
    output_file = "dataset_NT_with_scores_so.csv"
    df.to_csv(output_file, index=False)
    print(f"DataFrame successfully saved to {output_file}")

def download_nemotron_dataset(buffer_size=15000, total_sample=10000, split='vendor', filename="dataset/Nemotron_SFT_Science_v2.json"):
    ds = load_dataset(
        "nvidia/Nemotron-SFT-Science-v2",
        split,
        split="train",
        streaming=True
    )
    ds = ds.shuffle(seed=42, buffer_size=buffer_size)
    rows = [row for _, row in zip(range(total_sample), ds)]
    nemotron = []
    for idx, row in enumerate(rows):
        dic = {}
        dic['idx'] = idx + 1
        dic['uuid'] = row['uuid']
        dic['messages'] = row['messages']  
        dic['metadata'] = row['metadata']
        nemotron.append(dic)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(nemotron, f, indent=4, ensure_ascii=False)
    print(f"{filename} created.")

def get_topics(path):
    with open(path, 'r') as f:
        data = json.load(f)
    dic = {}
    for row in data:
        topic = row['metadata']['topic']
        if topic not in dic.keys():
            dic[topic] = 0
        dic[topic] += 1
    for topic in dic:
        print(f"Topic = {topic}: total_sample = {dic[topic]}")

def save_samples(input_path, output_path='dataset/Nemotron_SFT_Science_v2_10K.json'):
    with open(input_path, 'r') as f:
        ds = json.load(f)
    samples = []
    for idx, row in enumerate(ds):
        dic = {}
        dic['idx'] = idx
        dic['exercise'] = row['messages'][0]['content']
        dic['solution'] = row['messages'][1]['content']
        dic['topic'] = row['metadata']['topic']
        samples.append(dic)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=4)

def _is_bad_char(c: str) -> bool:
    cp = ord(c)
    if cp < 128:
        return False
    if 0x1D400 <= cp <= 0x1D7FF:
        return True
    cat = unicodedata.category(c)
    if cat in ("Mn", "Mc", "Me"):
        return True
    if cat in ("Cc", "Cf", "Cs", "Co"):
        return True
    if cat[0] == "L":
        try:
            name = unicodedata.name(c)
        except ValueError:
            return True
        if not name.startswith(_ALLOWED_LETTER_PREFIXES):
            return True
    return False

def is_clean(s: str) -> bool:
    return not any(_is_bad_char(c) for c in s)

def load_json_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_csv_data(path):
    df = pd.read_csv(path)
    return df

def get_topic_distribution(samples):
    dic = {}
    for sample in samples:
        if sample['topic'] not in dic.keys():
            dic[sample['topic']] = 0
        dic[sample['topic']] += 1
    print("-------------------- Eligible samples --------------------")
    for key in dic:
        print(f"Topic={key}: total_sample={dic[key]}")
    print()

def eligible_exercises(path, min_chars=50, max_chars=4000) -> list[str]:
    data = load_csv_data(path)
    pool = []
    for index, row in data.iterrows():
        exercise = row['prompt']
        exercise_len = len(exercise)
        if ((exercise_len >= min_chars and exercise_len <= max_chars) and is_clean(exercise)):
            pool.append(row)
    return pool

def random_sampling(samples):
    dic = {}
    for sample in samples:
        if sample['topic'] not in dic.keys():
            dic[sample['topic']] = []
        dic[sample['topic']].append(sample)
    idx = random.randint(1, 1000)  # 758
    idx = 758
    random.seed(idx)
    samples_64 = []
    for key in dic:
        candidates = dic[key]
        random.shuffle(candidates)
        if key == "Physics":
            samples_64 += candidates[:22]
        else:
            samples_64 += candidates[:21]
    return (idx, samples_64)

def get_exercise_with_solution(exercise, solution):
    text = exercise + f"""
\n\n
##The solution of the given exercise is:
{solution}
"""
    return text

def save_xlsx_file(samples, path, add_solution=False):
    results = []
    # subsets = []
    problem_ids = []
    for dic in samples:
        exercise = dic['prompt']
        solution = dic['Code']
        # topic = dic['topic'].strip()
        problem_id = dic['ProblemID']
        problem_ids.append(problem_id)
        exercise = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', exercise)
        solution = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', solution)
        if add_solution:
            text = get_exercise_with_solution(exercise, solution)
            results.append(text)
            # subsets.append(topic)
        else:
            results.append(exercise)
            # subsets.append(topic)
    data = {
        'Result': results,
        # 'Subject': subsets
        'ProblemID': problem_ids 
    }
    df1 = pd.DataFrame(data, columns=data.keys())
    df1.to_excel(path, index=False)
    print(f"Dataframe saved at {path}.")

if __name__ == "__main__":
    input_path='dataset/CW_random_50.csv'
    samples = eligible_exercises(input_path)
    # print(len(samples))

    # ## Random sampling
    # idx, sample_64 = random_sampling(samples) 
    # print(f"Seed={idx}: total_random_sample={len(sample_64)}")
    save_xlsx_file(samples, path = f'CW_Random_with_solution.xlsx', add_solution=True)
    