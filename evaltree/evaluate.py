"""In-pipeline evaluation for EvalTree hierarchies: entailment of sampled
descendant leaves against each internal node's summary, plus a specificity
check against the most-similar outsider leaf from sibling clusters."""

import random
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from evaltree.llm import generate_text
from evaltree.prompts import ENTAILMENT_PROMPT, ENTAILMENT_SYSTEM
from tree import Tree

_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-mpnet-base-v2')
    return _embed_model


def safe_extract_score(text):
    match = re.findall(r"Score:\s*(\d+(?:\.\d+)?)", text)
    if match:
        return float(match[0])
    return 0.0


def evaluate_node(child_prompt, parent_summary, model_name="openai", api_keys=None):
    """Scores how well a parent summary generalizes a child prompt."""
    if not child_prompt or not parent_summary:
        return 0.0, "Error: Missing input"
    prompt = ENTAILMENT_PROMPT.format(prompt_a=child_prompt, prompt_b=parent_summary)
    try:
        response = generate_text(prompt, ENTAILMENT_SYSTEM,
                                 model_name=model_name, api_keys=api_keys)
        return safe_extract_score(response), response
    except Exception as e:
        print(f"Error evaluating node: {e}")
        return 0.0, f"Error: {str(e)}"


def get_descendant_leaves(tree_levels, node_id, memo=None):
    if memo is None:
        memo = {}
    if node_id in memo:
        return memo[node_id]

    node = None
    node_level = -1
    for i, lvl in enumerate(tree_levels):
        for n in lvl.nodes:
            if n.id == node_id:
                node, node_level = n, i
                break
        if node:
            break
    if not node:
        return []

    if not node.children or node_level == 0:
        memo[node_id] = [node]
        return [node]

    leaves, seen = [], set()
    for cid in node.children:
        for leaf in get_descendant_leaves(tree_levels, cid, memo):
            if leaf.id not in seen:
                seen.add(leaf.id)
                leaves.append(leaf)
    memo[node_id] = leaves
    return leaves


def run_evaluation_on_tree(tree: Tree, num_eval_samples: int = 2,
                           model_name: str = "openai", api_keys: dict = None) -> None:
    """Populates entail_1 (mean over sampled leaves), entail_2 (worst sampled
    leaf), specificity, cluster_size, level_num and the evaluation text on
    every internal node."""
    print("\n--- Running EvalTree evaluation ---")
    random.seed(42)  # reproducible leaf sampling

    all_leaves = tree.levels[0].nodes if tree.levels else []
    memo = {}

    for level_idx in range(1, len(tree.levels)):
        level = tree.levels[level_idx]

        level_leaf_pools = {p.id: get_descendant_leaves(tree.levels, p.id, memo)
                            for p in level.nodes}

        for parent in level.nodes:
            scores, evals = [], []

            descendant_leaves = level_leaf_pools[parent.id]
            parent.cluster_size = len(descendant_leaves)
            parent.level_num = level_idx

            # 1. Entailment: sample descendant leaves against the summary
            if descendant_leaves:
                sampled = random.sample(descendant_leaves,
                                        min(num_eval_samples, len(descendant_leaves)))
                for i, child in enumerate(sampled):
                    child_text = child.summary or child.prompt
                    score, eval_text = evaluate_node(child_text, parent.summary,
                                                     model_name=model_name, api_keys=api_keys)
                    scores.append(score)
                    evals.append(f"Child {child.id} (Sample {i + 1}):\n{eval_text}")

            if scores:
                parent.entail_1 = sum(scores) / len(scores)
                parent.entail_2 = min(scores)

                # 2. Specificity: the sibling-cluster leaf closest to this
                # cluster's centroid should NOT be accepted by the summary.
                descendant_ids = {n.id for n in descendant_leaves}
                sibling_leaves = [leaf for sib in level.nodes if sib.id != parent.id
                                  for leaf in level_leaf_pools[sib.id]]

                seen, outsider_pool = set(), []
                for n in sibling_leaves:
                    if n.id not in descendant_ids and n.id not in seen:
                        seen.add(n.id)
                        outsider_pool.append(n)
                if not outsider_pool:
                    outsider_pool = [n for n in all_leaves if n.id not in descendant_ids]

                if outsider_pool:
                    embed_model = get_embed_model()
                    descendant_texts = [n.summary or n.prompt for n in descendant_leaves]
                    centroid = None
                    if descendant_texts:
                        vecs = embed_model.encode(descendant_texts, convert_to_numpy=True)
                        centroid = np.mean(vecs, axis=0)
                        centroid = centroid / (np.linalg.norm(centroid) or 1.0)

                    best_outsider, best_sim = None, -2.0
                    if centroid is not None:
                        for outsider in outsider_pool:
                            text = outsider.summary or outsider.prompt
                            if not text:
                                continue
                            vec = embed_model.encode(text, convert_to_numpy=True)
                            vec = vec / (np.linalg.norm(vec) or 1.0)
                            sim = float(np.dot(centroid, vec))
                            if sim > best_sim:
                                best_sim, best_outsider = sim, outsider
                    if best_outsider is None:
                        best_outsider = random.choice(outsider_pool)

                    outsider_text = best_outsider.summary or best_outsider.prompt
                    spec_score, spec_eval = evaluate_node(outsider_text, parent.summary,
                                                          model_name=model_name, api_keys=api_keys)
                    parent.specificity = spec_score
                    evals.append(f"\n--- SPECIFICITY (Sibling Outsider {best_outsider.id}) ---\n{spec_eval}")
                else:
                    parent.specificity = 0.0

            parent.evaluation = "\n\n".join(evals)
            print(f"Parent {parent.id} scored. Entail_1: {parent.entail_1}, "
                  f"Entail_2: {parent.entail_2}, Specificity: {parent.specificity}")
