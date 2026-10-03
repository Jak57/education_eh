# import os
import pickle
import heapq

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully.\n")
    return cache_data['sentences'], cache_data['embeddings']

def get_2d_matrix(n=50):
    # return [[False] * n] * n
    # return [[False] * n for _ in range(n)]
    return [[False] * n for _ in range(n)]

if __name__ == "__main__":
    path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    print(f"Loading embedding from {path} ...")

    sentences, embed = load_embeddings(path)
    # print("Embedding loaded successfully.\n")
    # print(type(sentences), len(sentences))
    # print(type(embed), len(embed), embed.shape)

    # print(sentences[0])
    # print(embed[0])

    similarity_matrix = embed @ embed.T
    # print(similarity_matrix.shape)
    # print("\nmatrix")
    # print(similarity_matrix)

    # print(get_2d_matrix(len(sentences)))

    # state_2d = get_2d_matrix(len(sentences))

    total_exercise = len(sentences)
    total_exercise = 5
    state2d = get_2d_matrix(total_exercise)

    # print(state_2d.shape)
    print(state2d)

    index_list = []
    hq = []
    for i in range(total_exercise):
        for j in range(total_exercise):
            if i != j:
                index_list.append((i, j))
                print(i, j, similarity_matrix[i][j])
                heapq.heappush(hq, (-similarity_matrix[i][j], (i, j)))
        print("----")

    print("Heap:", len(hq))
    link_count = [0] * total_exercise
    max_child = 3

    cnt = 0
    while len(hq) > 0:
        top = heapq.heappop(hq)
        _, index = top
        i, j = index

        print("\nNew item in heap:")
        print(top, index, i, j)
        # for k in range(len(state2d)):
        #     for l in range(len(state2d[0])):
        #         print(state2d[k][l], end=" ")
        #     print()

        # if index[0]
        if (not state2d[i][j]) and (link_count[i] < max_child):
            state2d[i][j] = True
            state2d[j][i] = True
            cnt += 1
            print("top --------------------------------------------->", top, cnt+1)
            link_count[i] += 1
            print(link_count)
            print()
            # print(state2d)
            # for k in range(len(state2d)):
            #     for l in range(len(state2d[0])):
            #         print(state2d[k][l], end=" ")
            #     print()
        # else:

        for k in range(len(state2d)):
            for l in range(len(state2d[0])):
                print(state2d[k][l], end=" ")
            print()

    print(cnt)






## python -m test.test_similarity_matrix