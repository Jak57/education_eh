import ast, json, io, requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from tree import Tree
# import unicodedata
import editdistance


def add_scores(tree, score_file):
    base_level = tree.levels[0].nodes
    score_db = pd.read_csv(score_file)
    instructions = score_db['instructions']
    score_cols = [c for c in score_db.columns if c.startswith("score_")]

    for node in tqdm(base_level, desc="Attaching scores to leaf nodes"):
        prompt = node.prompt
        min_dist = float('inf')
        min_scores = {col: None for col in score_cols}

        for idx, instruction in enumerate(instructions):
            dist = editdistance.eval(prompt, instruction)
            if dist < min_dist:
                min_dist = dist
                for col in score_cols:
                    min_scores[col] = score_db[col][idx]
                if min_dist == 0:
                    break

        node.wb_scores = {
            col: (float(v) if v is not None and not pd.isna(v) else 5.0)
            for col, v in min_scores.items()
        }
        # backward compat
        node.wb_score_mistral = node.wb_scores.get("score_Mistral-7B-Instruct-v0.2", 5.0)
        node.wb_score_llama   = node.wb_scores.get("score_Meta-Llama-3-8B-Instruct", 5.0)


def build_scores(tree):
    for i in range(1, len(tree.levels)):
        level = tree.levels[i].nodes
        prev_level = tree.levels[i - 1].nodes
        prev_by_id = {n.id: n for n in prev_level}

        for node in level:
            children = node.children
            if not children:
                continue

            sums: dict[str, float] = {}
            count = 0
            for child_id in children:
                child_node = prev_by_id.get(child_id)
                if child_node is None:
                    continue
                count += 1
                for col, val in child_node.wb_scores.items():
                    sums[col] = sums.get(col, 0.0) + val

            if count:
                node.wb_scores = {col: total / count for col, total in sums.items()}
                node.wb_score_mistral = node.wb_scores.get("score_Mistral-7B-Instruct-v0.2", 5.0)
                node.wb_score_llama   = node.wb_scores.get("score_Meta-Llama-3-8B-Instruct", 5.0)
    
    
# json_file = 'trees/uniform_64_haiku_sumv5_4o-mini_evalv1.json'

# tree_test = Tree.load(json_file)
# add_scores(tree_test, 'dataset_WB_with_scores.csv')
# build_scores(tree_test)
# tree_test.dump(json_file)