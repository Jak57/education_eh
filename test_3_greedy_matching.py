from test_2_simple_tree import Tree, Level, Node

def build_common_summary_tree(
        exercises: dict[int, str],
        summaries: dict[int, str],
        exercise_to_summary: dict[int, list[int]],
) -> Tree:
    # pass
    # Create 2 levels
    exercise_level = Level(level=0)
    summary_level = Level(level=1)

    # Reverse map
    summary_to_exercises: dict[int, list[int]] = {}
    for exercise_id, summary_ids in exercise_to_summary.items():
        for summary_id in summary_ids:
            if summary_id not in summary_to_exercises:
                summary_to_exercises[summary_id] = []
            if exercise_id not in summary_to_exercises[summary_id]:
                summary_to_exercises[summary_id].append(exercise_id)

    # Level 0: Create exercise nodes
    for exercise_id, prompt in exercises.items():
        summary_ids = exercise_to_summary.get(
            exercise_id,
            []
        )
        node = Node(
            id=exercise_id,
            node_type="exercise",
            level_num=0,
            prompt=prompt,
            parent=list(summary_ids),
            children=[],
            cluster_size=len(summary_ids),
        )
        exercise_level.nodes.append(node)

    # Level 1: Create common-summary nodes
    for summary_id, summary_text in summaries.items():
        exercise_ids = summary_to_exercises.get(
            summary_id,
            []
        )
        node = Node(
            id=summary_id,
            node_type="summary",
            level_num=1,
            capability=summary_text,
            summary=summary_text,
            prompt=summary_text,
            children=list(exercise_ids),
            parent=[],
            cluster_size=len(exercise_ids)
        )
        summary_level.nodes.append(node)
    # Determine next available ID
    all_ids = [
        node.id
        for node in exercise_level.nodes
    ]
    all_ids.extend(
        node.id
        for node in summary_level.nodes
    )
    next_id = (
        max(all_ids) + 1
        if all_ids
        else 0
    )
    # Create Tree
    tree = Tree(
        levels=[
            exercise_level,
            summary_level
        ],
        next_id=next_id,
        config={
            "method": "common_summary",
            "num_exercises": len(exercises),
            "num_summaries": len(summaries),
            "structure": "two_level_bipartite",
        }
    )
    return tree

if __name__ == "__main__":
    print("hello world")


# python test_3_greedy_matching.py