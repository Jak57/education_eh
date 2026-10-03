import pickle
import heapq

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully.\n")
    return cache_data['sentences'], cache_data['embeddings']

def get_similar_pairs(embedding_filepath, TOTAL_EXERCISE=10, LEVEL_1_NODE_INC_FACTOR=3):
    TOTAL_LEVEL_1_NODE = TOTAL_EXERCISE * LEVEL_1_NODE_INC_FACTOR
    TOTAL_NODE_TO_PAIR = 2 * LEVEL_1_NODE_INC_FACTOR
    print(f"Loading embedding from {embedding_filepath} ...")
    sentences, embed = load_embeddings(embedding_filepath)
    similarity_matrix = embed @ embed.T
    state2d = [[False] * TOTAL_EXERCISE for _ in range(TOTAL_EXERCISE)]
    hq = []
    for i in range(TOTAL_EXERCISE):
        for j in range(TOTAL_EXERCISE):
            if i != j:
                heapq.heappush(hq, (-similarity_matrix[i][j], (i, j)))
    print(f"From {TOTAL_EXERCISE} nodes, generating {TOTAL_LEVEL_1_NODE} pairs by associating each node with {TOTAL_NODE_TO_PAIR} other nodes...")
    link_count = [0] * TOTAL_EXERCISE
    candidateK_pair_list = []
    while len(hq) > 0:
        top = heapq.heappop(hq)
        _, index = top
        i, j = index
        if (not state2d[i][j]) and (link_count[i] < LEVEL_1_NODE_INC_FACTOR):
            state2d[i][j] = True
            state2d[j][i] = True
            candidateK_pair_list.append(index)
            link_count[i] += 1
    print(f"Pairs: {sorted(candidateK_pair_list)}")
    return sorted(candidateK_pair_list)

if __name__ == "__main__":
    path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    similar_pairs = get_similar_pairs(path, TOTAL_EXERCISE=10, LEVEL_1_NODE_INC_FACTOR=3)


## python -m test.test_similarity_matrix