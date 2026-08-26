"""Leaf annotation: one gerund-phrase capability per level-0 prompt."""

from evaltree.llm import generate_text
from evaltree.prompts import ANNOTATION_PROMPT, ANNOTATION_SYSTEM
from tree import Tree


def annotate_tree(tree: Tree, model_name: str = "claude", api_keys: dict = None,
                  force: bool = False) -> None:
    """Populates node.capability for every leaf. Skips already-annotated
    leaves unless force is set, so an interrupted run can resume."""
    leaves = tree.levels[0].nodes
    print(f"Annotating {len(leaves)} leaves...")
    for node in leaves:
        if node.capability and not force:
            continue
        prompt = ANNOTATION_PROMPT.format(instruction=node.prompt or "", response="N/A")
        try:
            capability = generate_text(prompt, ANNOTATION_SYSTEM,
                                       model_name=model_name, api_keys=api_keys)
            node.capability = capability.strip().strip('"')
            print(f"Node {node.id}: {node.capability}")
        except Exception as e:
            print(f"Error annotating node {node.id}: {e}")
            node.capability = "Unknown capability"
