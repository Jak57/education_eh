import pickle
import heapq

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully.\n")
    return cache_data['sentences'], cache_data['embeddings']

def get_similar_pairs(embedding_filepath, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3):
    TOTAL_LEVEL_1_NODE = TOTAL_EXERCISE * LEVEL_1_NODE_INC_FACTOR
    TOTAL_NODE_TO_PAIR = 2 * LEVEL_1_NODE_INC_FACTOR
    print(f"Loading embedding from {embedding_filepath} ...")
    sentences, embed = load_embeddings(embedding_filepath)
    similarity_matrix = embed @ embed.T

    sentence_id_map = {}
    for i in range(len(sentences)):
        sentence_id_map[str(i)] = sentences[i]
    state2d = [[False] * TOTAL_EXERCISE for _ in range(TOTAL_EXERCISE)]
    hq = []
    for i in range(TOTAL_EXERCISE):
        for j in range(TOTAL_EXERCISE):
            if i != j:
                heapq.heappush(hq, (-similarity_matrix[i][j], (i, j)))
    print(f"From {TOTAL_EXERCISE} nodes, generating {TOTAL_LEVEL_1_NODE} pairs by associating each node with {LEVEL_1_NODE_INC_FACTOR} other nodes...")
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
    print(f"\nTotal pairs={len(candidateK_pair_list)}")
    return sentence_id_map, sorted(candidateK_pair_list)

def get_pair_text(path, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3):
    text_pair = []
    sentence_id_map, similar_pairs = get_similar_pairs(path, TOTAL_EXERCISE, LEVEL_1_NODE_INC_FACTOR)
    for index in similar_pairs:
        text1 = sentence_id_map[str(index[0])]
        text2 = sentence_id_map[str(index[1])]
        text_pair.append((text1, text2))
    return text_pair

if __name__ == "__main__":
    path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    text_pair = get_pair_text(path, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3)
    print(len(text_pair))
    print(text_pair[0])

## python -m test.test_similarity_matrix