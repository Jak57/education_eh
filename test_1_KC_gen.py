from pathlib import Path
import os

# from test.test_similarity_matrix import save_mapping
from test_4_similarity_matrix import save_mapping, get_mapping
from test_3_greedy_matching import build_common_summary_tree


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

# def get_pro

if __name__ == "__main__":
    embedding_path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    path = "outputs/text_summary_map_150.csv"
    # save_mapping(embedding_path, api_keys=API_KEYS, output_path=path, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3)

    exercises, summaries, exercise_to_summaries = get_mapping(path, TOTAL_EXERCISE=50)
    print(len(exercises), len(summaries), len(exercise_to_summaries))

    tree = build_common_summary_tree(exercises, summaries, exercise_to_summaries)
    tree.dump("test_tree.json")


## python test_1_KC_gen.py