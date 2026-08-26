import ast, json, io, requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# ------------------------------------------------------------------
# 1. Load the WildBench v2 "test" split
# ------------------------------------------------------------------
# ds = load_dataset("openbmb/UltraFeedback")
ds = load_dataset("allenai/WildBench", "v2", split="test")

df = pd.DataFrame(ds)

# ------------------------------------------------------------------
# 2. Discover all GPT-4o evaluation-result JSONs from the WildBench
#    GitHub directory listing and build download URLs dynamically.
# ------------------------------------------------------------------
LISTING_URL = (
    "https://api.github.com/repos/allenai/WildBench/contents/"
    "eval_results/v2.0625/score.v2/eval%3Dgpt-4o-2024-05-13"
)
listing = requests.get(LISTING_URL, timeout=30).json()
SCORE_FILES = [f["download_url"] for f in listing if f["name"].endswith(".json")]
print(f"[INFO] Found {len(SCORE_FILES)} model score files to merge.")

# ------------------------------------------------------------------
# 3. Helper to download & parse one score file → DataFrame
# ------------------------------------------------------------------
def load_score_file(url: str) -> pd.DataFrame:
    """
    Return a two-column DataFrame: session_id plus a model-specific score.
    Column is renamed to score_<model_name> so multiple files can coexist.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = json.loads(resp.text)

    score_df = pd.DataFrame(data)[["session_id", "score"]]
    model_name = url.split("/")[-1].replace(".json", "")
    score_df = score_df.rename(columns={"score": f"score_{model_name}"})
    return score_df

# ------------------------------------------------------------------
# 4. Extract user prompts from the conversation_input field
# ------------------------------------------------------------------
def extract_content(conversation):
    try:
        if isinstance(conversation, str):
            conversation = ast.literal_eval(conversation)

        # keep only user messages
        data = [m["content"] for m in conversation if m["role"] == "user"]

        # normalise to a flat list of strings
        if isinstance(data, list):
            msgs = data
        elif isinstance(data, str):
            try:
                msgs = ast.literal_eval(data)
            except Exception:
                msgs = [data]
        else:
            msgs = list(data)

        return "".join(msgs)
    except Exception:
        return ""




df["instructions"] = df["conversation_input"].apply(extract_content)

# ------------------------------------------------------------------
# 5. Drop unneeded columns
# ------------------------------------------------------------------
df = df.drop(
    [
        # "id",
        "conversation_input",
        "references",
        "length",
        "checklist",
        "intent",
        "secondary_tags",
    ],
    axis=1,
)






# ------------------------------------------------------------------
# 6. Merge each score file on session_id
# ------------------------------------------------------------------
for url in SCORE_FILES:
    score_df = load_score_file(url)
    df = df.merge(score_df, on="session_id", how="left")

# ------------------------------------------------------------------
# 7. Write the enriched CSV
# ------------------------------------------------------------------
output_file = "dataset_WB_with_scores.csv"
df.to_csv(output_file, index=False)
print(f"DataFrame successfully saved to {output_file}")


