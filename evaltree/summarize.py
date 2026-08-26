"""Cluster summaries: one broad gerund phrase per internal node, generated
bottom-up so each parent summarizes its children's summaries/capabilities."""

from evaltree.llm import generate_text
from evaltree.prompts import SUMMARY_PROMPT, SUMMARY_SYSTEM
from tree import Tree


def generate_cluster_summary(descriptions, model_name: str = "claude",
                             api_keys: dict = None) -> str:
    """One gerund phrase summarizing a list of capability descriptions.
    Retries when the model copies an input verbatim."""
    if not descriptions:
        return "Unknown capability"
    desc_str = "\n".join(f"- {d}" for d in descriptions)
    prompt = SUMMARY_PROMPT.format(descriptions=desc_str)
    system = (SUMMARY_SYSTEM + "\nCRITICAL: DO NOT copy the input descriptions "
              "verbatim. You must synthesize them into a new gerund phrase.")

    final_summary = "Unknown capability"
    for attempt in range(3):
        try:
            summary = generate_text(prompt, system, model_name=model_name,
                                    api_keys=api_keys).strip().strip('"')
            lowered = summary.lower()
            if any(lowered in (d.strip().lower(), f"- {d.strip().lower()}")
                   for d in descriptions):
                raise ValueError("Verbatim copy detected.")
            final_summary = summary
            break
        except Exception as e:
            print(f"Error generating cluster summary (attempt {attempt + 1}): {e}")
    return final_summary


def summarize_tree(tree: Tree, model_name: str = "claude", api_keys: dict = None) -> None:
    """Populates summary (and prompt, which downstream code reads) for every
    internal node, level by level from the bottom up."""
    node_map = {n.id: n for lvl in tree.levels for n in lvl.nodes}
    for lvl in tree.levels[1:]:
        for node in lvl.nodes:
            child_texts = []
            for cid in node.children:
                child = node_map.get(cid)
                if child is None:
                    continue
                text = child.summary or child.capability or child.prompt
                if text:
                    child_texts.append(text)
            if not child_texts:
                node.summary = "Unknown capability"
                continue
            node.summary = generate_cluster_summary(child_texts, model_name=model_name,
                                                    api_keys=api_keys)
            node.prompt = node.summary
            print(f"Node {node.id} (L{lvl.level}) summary: {node.summary}")
