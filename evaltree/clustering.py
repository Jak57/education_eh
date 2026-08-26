"""Recursive 2-means clustering over leaf embeddings, and conversion of the
nested clustering into the levels-based Tree structure."""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from tree import Level, Node, Tree


class RecursiveKMeans:
    def __init__(self, max_children=2, min_samples=1, max_depth=50):
        self.max_children = max_children
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, texts):
        return self.model.encode(texts)

    def find_optimal_k(self, embeddings):
        if len(embeddings) <= 1:
            return 1
        return 2

    def cluster_recursive(self, node_ids, embeddings, depth=0):
        """Returns a nested dict {"ids": [...], "children": [...]}."""
        if depth >= self.max_depth or len(node_ids) <= self.min_samples:
            return {"ids": node_ids, "children": []}

        k = self.find_optimal_k(embeddings)
        if k <= 1:
            return {"ids": node_ids, "children": []}

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        splits = []
        for i in range(k):
            mask = (labels == i)
            child_ids = [node_ids[j] for j, m in enumerate(mask) if m]
            if child_ids:
                splits.append((child_ids, embeddings[mask]))

        # Degenerate split: KMeans put every point in one cluster (happens when
        # the points are identical, e.g. duplicate leaf capability annotations).
        # Recursing would rebuild the identical set until max_depth, producing a
        # chain of unary pass-through nodes.
        if len(splits) <= 1:
            return {"ids": node_ids, "children": []}

        children = [self.cluster_recursive(child_ids, child_embeddings, depth + 1)
                    for child_ids, child_embeddings in splits]
        return {"ids": node_ids, "children": children}


def build_hierarchy(tree: Tree, clustering_result: dict) -> None:
    """Turns the nested clustering dict into internal Nodes and rebuilds
    tree.levels so that leaves sit at level 0 and every internal node sits at
    its height (longest path down to a leaf). Mutates the tree in place."""
    leaves = tree.levels[0].nodes
    node_map = {n.id: n for n in leaves}
    internal_nodes = []

    def build_flat(cluster_dict):
        if not cluster_dict["children"]:
            return cluster_dict["ids"]
        all_child_ids = []
        for child in cluster_dict["children"]:
            all_child_ids.extend(build_flat(child))
        parent = Node(id=tree.new_id(), children=all_child_ids)
        for cid in all_child_ids:
            if cid in node_map:
                node_map[cid].parent.append(parent.id)
        node_map[parent.id] = parent
        internal_nodes.append(parent)
        return [parent.id]

    build_flat(clustering_result)

    heights = {n.id: 0 for n in leaves}

    def get_height(node_id):
        if node_id in heights:
            return heights[node_id]
        node = node_map[node_id]
        if not node.children:
            heights[node_id] = 0
            return 0
        h = max(get_height(cid) for cid in node.children) + 1
        heights[node_id] = h
        return h

    for nid in node_map:
        get_height(nid)

    max_h = max(heights.values())
    tree.levels = [
        Level(level=h, nodes=[node_map[nid] for nid, hh in heights.items() if hh == h])
        for h in range(max_h + 1)
    ]
