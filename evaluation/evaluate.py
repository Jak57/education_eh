"""Post-hoc faithfulness + informativeness evaluation for a built tree.

Merges the two standalone evaluators that previously lived in separate
scripts (evaluate_final_json.py for faithfulness, evaluate_final_info_json.py
for informativeness) into one pass. Works on any levels-format tree JSON —
blossom or evaltree — and writes each metric under its own field name
(faithfulness1/2, informativeness1/2, specificity_faithfulness,
specificity_informativeness, plus the evaluation_* text fields), so a single
output file carries every metric and stats.py can read it directly.

Per internal node: two descendant leaves are sampled once and scored with
both prompts (the same leaves for both metrics, so the numbers are
comparable). For the two specificity variants the outsider is the
sibling-cluster leaf whose embedding is closest to the node's descendant-leaf
centroid (recorded as third_leaf / third_leaf_sim) — a summary that accepts
its nearest outsider is insufficiently specific.

Usage (from the repository root):
    python evaluation/evaluate.py <tree.json> [more.json ...]
    python evaluation/evaluate.py <tree.json> --model openai --samples 2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.simple_eval import _judge_call, safe_extract_score  # noqa: E402


def _load_env() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        print(f"[WARN] no .env found at {path} -- API keys will be empty")
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


FAITHFULNESS_PROMPT = """
You will be given a prompt A (child) and a general prompt B (parent) that is meant to be a generalization of prompt A. Your task is to determine how FAITHFUL prompt B is to prompt A.

Faithfulness measures whether everything prompt B asks for is actually grounded in prompt A. Calculate a score between 0 and 1 for what fraction of the distinct details in prompt B (the parent) are present in, or implied by, prompt A (the child).

Important:
- This is a check for the ABSENCE of hallucination. A faithful parent introduces no task, domain, constraint, or detail that prompt A does not support.
- Do NOT penalize abstraction or omission. It is expected and correct for B to drop specifics from A (e.g., specific libraries, datasets, or minor constraints) — even to keep only A's topic or domain and drop A's task or format. Leaving things out never lowers faithfulness — a parent that drops detail but invents nothing is fully faithful (1.0) — and "B omits A's requirements" is an informativeness concern, not grounds to lower this score. This protects only what B keeps: B is still penalized as usual for any detail it states that A does not support.
- ONLY penalize content in B that does not come from A — constraints B invents, or a task/domain B assigns that A is not actually about.
- Implicit support counts fully. A detail in B does NOT need to appear verbatim in A — if A clearly implies it, treat it as supported and do not deduct for it. Never lower the score just because A does not state B's wording explicitly. An abstract summary whose details are all implied (not stated) by A is well-grounded (0.6 and above), not a mismatch.
- Implication can be partial. If a detail in B is typically but not necessarily implied by A (it usually follows but is not guaranteed), give it partial credit rather than full credit, rather than treating it as fully grounded.
- A very broad or vague parent can still be faithful, as long as the little it does say is supported by the child.

Base your score on the ratio of B's distinct details that are supported by A, out of the total distinct details in B.

Scoring guidance:
- 0.85-1.0 (Fully Grounded): Every detail, task, and constraint in Prompt B is present in or directly implied by Prompt A. B may be far more abstract than A, but it adds nothing A does not support.
- 0.6-0.85 (Mostly Grounded): Nearly all of B is supported by A — explicitly or implicitly — with only a minor unsupported nuance, a slight shift in framing, or a detail that is typically but not necessarily implied by A. An abstract summary whose details are all implied by A (even if never stated verbatim) belongs here.
- 0.3-0.6 (Partially Grounded): B mixes supported content with a noticeable detail or constraint that A does not support even implicitly. (Reserve this for genuinely absent or invented content — not for details that A merely implies rather than states.)
- 0.0-0.3 (Hallucinated / Mismatched): B introduces a task, domain, or constraint that is absent from or contradicts A (e.g., A trains an ML model, but B describes user-facing CRUD operations).

## Examples

Prompt A (child): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (parent): Write a Python program to train a deep learning model for image classification.
Score: 1.0
Reason: Every claim in B — Python, training a deep learning model, image classification — is fully present in A. B abstracts away CNNs, MNIST, and Matplotlib, but omission is never penalized, and B invents nothing, so faithfulness is perfect.

Prompt A (child): Develop a deep reinforcement learning agent. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (parent): Writing code.
Score: 0.75
Reason: B's only claim is "writing code," which is usually — but not strictly — implied by developing a reinforcement-learning agent (one could develop an agent conceptually, or with no-code/configuration tools). Because the detail is probable rather than guaranteed, it is mostly grounded rather than fully grounded. (B is also extremely vague, but that is an informativeness issue, not a faithfulness one.)

Prompt A (child): Write a Python web scraper using requests and BeautifulSoup that collects book titles and prices from an online bookstore's catalog pages and writes them to a CSV file.
Prompt B (parent): Write a Python program that gathers information from a website and saves it for later use.
Score: 0.7
Reason: None of B's wording appears verbatim in A, but A clearly implies all of it: "gathers information from a website" is exactly what the scraper does, and "saves it" is implied by writing to a CSV — implicit support counts, so these are grounded, not absent. "For later use" is a probable but not guaranteed purpose, so it earns partial rather than full credit. B is an abstract but well-grounded generalization, so it is mostly grounded — not the 0.3 a strict "not stated explicitly" reading would give.

Prompt A (child): Write a 1,000-word op-ed arguing that city councils should expand protected bike lanes, citing reduced traffic fatalities and lower carbon emissions.
Prompt B (parent): Write an opinion piece that argues for a local policy change and backs it up with concrete benefits.
Score: 0.65
Reason: Every detail of B is implied by A rather than stated outright: "opinion piece" covers an op-ed, "argues for a local policy change" is implied by advocating that city councils expand bike lanes, and "concrete benefits" is implied by citing fewer fatalities and lower emissions. Implicit support counts, so none of this is invented. Only a small penalty applies — for the abstraction and for "concrete benefits" being a loose generalization of A's specific reasons — leaving it mostly grounded rather than a 0.3.

Prompt A (child): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (parent): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.30
Reason: "Processes structured data" is loosely supported, but "implementing user-facing operations" is a hallucinated constraint — A trains a model and has no user-facing/CRUD component. The low score is for this invented task, NOT for B dropping A's specifics (BERT, sentiment analysis, F1) — that omission alone would not lower faithfulness.

## Prompt A (child)
{prompt_a}

## Prompt B (parent)
{prompt_b}

Output your score in the following format:
Score: [SCORE]
"""

INFORMATIVENESS_PROMPT = """
You will be given a prompt A (child) and a general prompt B (parent) that is meant to be a generalization of prompt A. Your task is to determine how INFORMATIVE prompt B is about prompt A.

Informativeness measures whether prompt B retains the important content of prompt A. Calculate a score between 0 and 1 for what fraction of the important contents of prompt A (the child) are present in, or implied by, prompt B (the parent).

Important:
- First identify the IMPORTANT contents of A: its core intent, domain, primary task, and key constraints. Ignore trivial or incidental details.
- We are looking for SEMANTIC COVERAGE, not strict detail preservation. If B correctly identifies the type of task and domain of A (e.g., "write an analytical essay about climate policy" or "code a data-processing script"), it should score highly even if it abstracts away specific tools or minor constraints.
- Penalize VAGUENESS: a parent so abstract that it gives almost no structural or topical guidance about A (e.g., "writing code", "analyzing data") is uninformative, even if technically true.
- Do NOT reward B for content not in A; only the important content of A that B actually captures counts.

Base your score on the ratio of A's important contents that are captured by B, out of A's total important contents.

Scoring guidance:
- 0.85-1.0 (Excellent Coverage): B captures essentially all of A's important content — its domain, primary task, and key constraints — dropping only minor specifics (a particular library, dataset, or metric).
- 0.6-0.85 (Good Coverage): B captures A's core domain and primary task but loses several important specifics or constraints (e.g., keeps "train a deep learning model for image classification" but drops "from scratch", the dataset, and the visualization requirement).
- 0.3-0.6 (Vague / Gerund Phrase): B is extremely abstract (e.g., "Analyzing data", "Writing content"). It captures the broad activity type but almost none of A's structural or topical content.
- 0.0-0.3 (Uninformative / Mismatched): B captures essentially none of A's important content, because it is far too generic or it miscategorizes the task.

## Examples

Prompt A (child): Implement a convolutional neural network (CNN) from scratch using PyTorch to classify handwritten digits from MNIST, with custom layers, training/validation loops without torchvision.models, and loss/accuracy curves in Matplotlib.
Prompt B (parent): Write a Python program to train a deep learning model for image classification.
Score: 0.75
Reasoning (for your understanding only): B captures A's core content — Python, training a deep learning model, image classification — but loses several important specifics: the "from scratch" constraint, the MNIST dataset, and the visualization requirement. Good coverage of the core, not excellent.

Prompt A (child): Develop a deep reinforcement learning agent using Python and TensorFlow 2. Implement the Deep Q-Network (DQN) algorithm to solve the CartPole-v1 environment from OpenAI Gym.
Prompt B (parent): Writing code.
Score: 0.15
Reasoning (for your understanding only): B conveys none of A's important content — not the reinforcement-learning domain, the DQN algorithm, TensorFlow, or CartPole. Technically true, but uninformative.

Prompt A (child): Write a Python script that uses Hugging Face Transformers to fine-tune a BERT model for sentiment analysis, including preprocessing, tokenization, a Trainer training loop, and accuracy/F1 evaluation.
Prompt B (parent): Write a program that processes and manages structured data, implementing user-facing operations with organized output.
Score: 0.15
Reasoning (for your understanding only): B miscategorizes A. It captures none of A's important content — machine learning, fine-tuning BERT, sentiment analysis, the NLP domain — describing a different kind of program entirely.

## Prompt A (child)
{prompt_a}

## Prompt B (parent)
{prompt_b}

Output your score in the following format:
Score: [SCORE]
"""


_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('all-mpnet-base-v2')
    return _embed_model


def _get_score(prompt_template, prompt_a, prompt_b, model_name, api_keys):
    user_prompt = prompt_template.format(prompt_a=prompt_a, prompt_b=prompt_b)
    for attempt in range(3):
        try:
            response = _judge_call(user_prompt, model_name=model_name, api_keys=api_keys)
            return safe_extract_score(response), response
        except Exception as e:
            print(f"API Error (attempt {attempt + 1}): {e}")
    return 0.0, "Error: all attempts failed"


def _node_text(node):
    return node.get("prompt") or node.get("summary") or ""


def evaluate_tree_file(input_path, model_name="openai", api_keys=None,
                       num_samples=2, seed=42, output_path=None):
    input_path = Path(input_path)
    if output_path is None:
        out_dir = input_path.parent / "eval_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / (input_path.stem + "_eval.json")

    random.seed(seed)
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    levels = data.get("levels", [])
    id2node = {n["id"]: n for lvl in levels for n in lvl.get("nodes", [])}

    def get_leaves(node_id):
        node = id2node[node_id]
        if not node.get("children"):
            return [node_id]
        leaves = []
        for child_id in node["children"]:
            leaves.extend(get_leaves(child_id))
        return list(set(leaves))

    # normalized embeddings for every leaf with text; the specificity
    # outsider is picked by cosine similarity against these
    all_leaf_ids = [n["id"] for n in levels[0]["nodes"]] if levels else []
    leaf_nodes = [id2node[lid] for lid in all_leaf_ids if _node_text(id2node[lid])]
    leaf_pos = {node["id"]: idx for idx, node in enumerate(leaf_nodes)}
    leaf_vecs = None
    if leaf_nodes:
        leaf_vecs = get_embed_model().encode([_node_text(n) for n in leaf_nodes],
                                             convert_to_numpy=True)
        norms = np.linalg.norm(leaf_vecs, axis=1, keepdims=True)
        leaf_vecs = leaf_vecs / np.maximum(norms, 1e-12)

    def find_outsider_leaf(node, level_nodes, level_leaf_pools):
        """The non-descendant leaf (sibling clusters first, any leaf as
        fallback) most similar to this node's descendant-leaf centroid."""
        if leaf_vecs is None:
            return None, None

        desc = set(level_leaf_pools[node["id"]])
        desc_pos = [leaf_pos[lid] for lid in desc if lid in leaf_pos]
        if not desc_pos:
            return None, None

        outsider_ids = set()
        for sibling in level_nodes:
            if sibling["id"] != node["id"]:
                outsider_ids.update(level_leaf_pools[sibling["id"]])
        outsider_ids -= desc
        if not outsider_ids:
            outsider_ids = set(all_leaf_ids) - desc

        outsider_pos = [leaf_pos[lid] for lid in outsider_ids if lid in leaf_pos]
        if not outsider_pos:
            return None, None

        centroid = leaf_vecs[desc_pos].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        sims = leaf_vecs[outsider_pos] @ centroid
        pick_pos = outsider_pos[int(np.argmax(sims))]
        return leaf_nodes[pick_pos], float(leaf_vecs[pick_pos] @ centroid)

    total = sum(len(lvl["nodes"]) for lvl in levels[1:])
    done = 0

    for lvl in levels:
        if lvl.get("level", 0) == 0:
            for node in lvl["nodes"]:
                node["cluster_size"] = 1
                node["level_num"] = 0
            continue

        lvl_nodes = lvl["nodes"]
        level_leaf_pools = {node["id"]: get_leaves(node["id"]) for node in lvl_nodes}
        for node in lvl_nodes:
            leaves = level_leaf_pools[node["id"]]
            node["cluster_size"] = len(leaves)
            node["level_num"] = lvl["level"]
            parent_text = _node_text(node)

            # sample once; the same leaves are scored with both prompts
            if len(leaves) >= num_samples:
                sampled = random.sample(leaves, num_samples)
            elif leaves:
                sampled = [leaves[0]] * num_samples
            else:
                sampled = []

            faith_scores, info_scores = [], []
            faith_evals, info_evals = [], []
            for leaf_id in sampled:
                leaf_text = _node_text(id2node[leaf_id])
                score, text = _get_score(FAITHFULNESS_PROMPT, leaf_text, parent_text,
                                         model_name, api_keys)
                faith_scores.append(score)
                faith_evals.append(f"Child {leaf_id}:\n{text}")
                score, text = _get_score(INFORMATIVENESS_PROMPT, leaf_text, parent_text,
                                         model_name, api_keys)
                info_scores.append(score)
                info_evals.append(f"Child {leaf_id}:\n{text}")

            if faith_scores:
                node["faithfulness1"] = sum(faith_scores) / len(faith_scores)
                node["faithfulness2"] = min(faith_scores)
                node["informativeness1"] = sum(info_scores) / len(info_scores)
                node["informativeness2"] = min(info_scores)
                node["evaluation_faithfulness"] = "\n\n".join(faith_evals)
                node["evaluation_informativeness"] = "\n\n".join(info_evals)
            else:
                node["faithfulness1"] = node["faithfulness2"] = None
                node["informativeness1"] = node["informativeness2"] = None

            outsider_leaf, outsider_sim = find_outsider_leaf(node, lvl_nodes, level_leaf_pools)
            if outsider_leaf is not None:
                outsider_text = _node_text(outsider_leaf)
                node["third_leaf"] = outsider_leaf["id"]
                node["third_leaf_sim"] = outsider_sim
                score, text = _get_score(FAITHFULNESS_PROMPT, outsider_text, parent_text,
                                         model_name, api_keys)
                node["specificity_faithfulness"] = score
                node["evaluation_specificity_faithfulness"] = text
                score, text = _get_score(INFORMATIVENESS_PROMPT, outsider_text, parent_text,
                                         model_name, api_keys)
                node["specificity_informativeness"] = score
                node["evaluation_specificity_informativeness"] = text
            else:
                node["specificity_faithfulness"] = None
                node["specificity_informativeness"] = None

            done += 1
            if done % 10 == 0:
                print(f"Evaluated {done}/{total} parents...")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", help="Tree JSONs (blossom or evaltree)")
    parser.add_argument("--model", default="openai",
                        choices=["openai", "claude", "gemma", "gemini"],
                        help="Judge provider (same options as MODEL_CONFIG)")
    parser.add_argument("--samples", type=int, default=2,
                        help="Descendant leaves sampled per parent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _load_env()
    api_keys = {
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
        "gemma": {
            "api_key": os.environ.get("GEMMA_API_KEY", "DUMMY"),
            "base_url": os.environ.get("GEMMA_BASE_URL", "http://<GPU_HOST>:8000/v1"),
            "model": os.environ.get("GEMMA_MODEL", "gemma-3-finetuned-merged-bf16"),
        },
        "gemini": {
            "api_key": os.environ.get("GEMINI_API_KEY", ""),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        },
    }

    for path in args.files:
        evaluate_tree_file(path, model_name=args.model, api_keys=api_keys,
                           num_samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
