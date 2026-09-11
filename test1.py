from pathlib import Path
import argparse
import json
import os
import pandas as pd

def _load_env(path: str = "") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments.
    Resolves .env next to this file, so it works regardless of the cwd."""
    if not path:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        print(f"[WARN] no .env found at {path} -- API keys will be empty")
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


_load_env()
# from common_summary_generation.summarize import generate_summary, get_entail_2_score
# from evaluation.evaluate import evaluate_tree_file
# from evaluation.simple_eval import evaluate_common_summary, promote_to_next_level, propagate_attrs_up, record_stat, reassign
# from tree import Tree, Level
# from evaltree.annotate import annotate_tree
# from evaltree.clustering import RecursiveKMeans, build_hierarchy
# from evaltree.summarize import summarize_tree
# from evaltree.evaluate import run_evaluation_on_tree as run_evaltree_evaluation
# from retrieval.blossom_matching import get_nearest_neighbors_blossom, _extract_main_idea_llm
# from retrieval.third_neighbor import find_third
# from visualisation.visualise_helper import (
#     generate_dag_json_from_tree,
#     generate_json_from_tree,
#     launch_tree_viewer,
# )
# from score_retrieval import (
#     add_scores,
#     build_scores
# )

from test.prompt import get_random_sample_pair, get_summary, print_summary
from utils import load_xlsx
from retrieval.blossom_matching import _extract_main_idea_llm

# --------------------------------------------------------------------------- #
# DATA_FILE = Path("Results/UF_random_prompts_64_base.json")
# SCORE_FILE = "dataset_UF_with_scores.csv"   # for UltraFeedback: dataset_UF_with_scores.csv


# ------------------------------ WildBench -----------------------------------#
## Tree json and Score file paths: Disimilar
# tree_json_filepath = "Results/Gemma/FINAL_random_prompts_64.json"
# score_filepath = "dataset_WB_with_scores.csv"


# ------------------------------ Nemotron-SFT-Science ------------------------#
## Tree json and Score file paths: Disimilar
tree_json_filepath = "Results/Gemma/NT_random_prompts_64.json"
score_filepath = "dataset_NT_with_scores_so.csv"

DATA_FILE = Path(tree_json_filepath)
SCORE_FILE = score_filepath

VISUALISE = True                 # flip to False to skip viewer step
MODEL_CONFIG = {
    "blossom": "claude",  # Options: "openai", "claude", "gemma", "gemini"
    "summary_generation": "claude",
    "evaluation": "openai"  # Options: "openai", "claude", "gemma", "gemini"
}
# Credentials come from .env (see .env.example); never hardcode keys here.
API_KEYS = {
    "openai": os.environ.get("OPENAI_API_KEY", ""),
    "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gemma": {
        "api_key": os.environ.get("GEMMA_API_KEY", "DUMMY"),
        "base_url": os.environ.get("GEMMA_BASE_URL", "http://<GPU_HOST>:8000/v1"),
        # must match the vLLM server's registered name (its --model path,
        # unless --served-model-name is set)
        "model": os.environ.get("GEMMA_MODEL", "gemma-3-finetuned-merged-bf16"),
    },
    "gemini": {
        "api_key": os.environ.get("GEMINI_API_KEY", ""),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}


# def load_exercises(path):

#     pass

if __name__ == "__main__":
    api_keys = API_KEYS
    path = "NT_Random_758_with_solution.xlsx"

    # print_summary(path, api_keys)

    exercises = load_xlsx(path)
    print(len(exercises))
    print(exercises[0])

    out = _extract_main_idea_llm(exercises)
    print(len(out))
    print(out[0])

    print((out == exercises))

