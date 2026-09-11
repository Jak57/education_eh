from test.prompt_template.common_summary import get_common_summary_prompt, get_system_prompt
import pandas as pd
import random
import anthropic
import re
import os

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

def read_excel_file(path):
    df = pd.read_excel(path)
    dic = {}
    for index, row in df.iterrows():
        dic[f"{index + 2}"] = row['Result']
    return dic

def get_random_sample_pair(path):
    dic = read_excel_file(path)
    idx1 = random.randint(2, 65)
    idx2 = random.randint(2, 65)
    print(f"Random sample ids={idx1} and {idx2}")
    exercise1_with_sol = dic[str(idx1)]
    exercise2_with_sol = dic[str(idx2)]
    idx_1 = exercise1_with_sol.find("##The solution of the given exercise is:")
    idx_2 = exercise2_with_sol.find("##The solution of the given exercise is:")
    return (exercise1_with_sol, exercise1_with_sol[:idx_1], exercise2_with_sol, exercise2_with_sol[:idx_2])

def get_summary(story1, story2,  model_name='claude', api_keys=None):
    system_prompt = get_system_prompt()
    prompt14 = get_common_summary_prompt(story1, story2)
    if model_name == "claude":
            client = anthropic.Anthropic(
                api_key=api_keys["claude"]
            )
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = message.content[0].text.strip()
            match = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
            summary = match.group(1).strip() if match else raw.strip()
            return summary
    return ""

def print_summary(path, api_keys):
    a, b, c, d = get_random_sample_pair(path)
    summary1 = get_summary(a, c, api_keys=api_keys)
    summary2 = get_summary(b, d, api_keys=api_keys)

    print("Summary for exercises with solutions:")
    print(summary1)
    print("\nSummary for exercises without solutions:")
    print(summary2)