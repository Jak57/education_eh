import pickle
import heapq
import pandas as pd

from test.prompt import get_summary
from utils import load_csv

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully.\n")
    return cache_data['sentences'], cache_data['embeddings']

def get_similar_pairs(embedding_filepath, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3):
    TOTAL_LEVEL_1_NODE = TOTAL_EXERCISE * LEVEL_1_NODE_INC_FACTOR
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

def get_pair_text(path, TOTAL_EXERCISE, LEVEL_1_NODE_INC_FACTOR):
    text_pair = []
    sentence_id_map, similar_pairs = get_similar_pairs(path, TOTAL_EXERCISE, LEVEL_1_NODE_INC_FACTOR)
    for index in similar_pairs:
        text1 = sentence_id_map[str(index[0])]
        text2 = sentence_id_map[str(index[1])]
        text_pair.append((text1, text2))
    return text_pair

def save_mapping(path, api_keys, output_path, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3):
    text_pair = get_pair_text(path, TOTAL_EXERCISE, LEVEL_1_NODE_INC_FACTOR)
    exercise1 = []
    exercise2 = []
    summary = []
    for i in range(len(text_pair)):
        text1, text2 = text_pair[i][0], text_pair[i][1]
        common_summary = get_summary(text1, text2, model_name='claude', api_keys=api_keys)
        exercise1.append(text1)
        exercise2.append(text2)
        summary.append(common_summary)
    data = {
        'exercise1': exercise1,
        'exercise2': exercise2,
        'common_summary': summary
    }
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(output_path, index=False)
    print(f"Output file saved at {output_path}")

def get_id(dic, text):
    for key in dic:
        if dic[key] == text:
            return key
    return None

def get_mapping(path, TOTAL_EXERCISE=50):
    df = load_csv(path)
    exercises = {}
    summaries = {}
    exercise_to_summaries = {}
    exercise_summary_dic = {}
    all_exercises = set()
    idx = 0
    summary_idx = TOTAL_EXERCISE
    for index, row in df.iterrows():
        exercise1 = row['exercise1']
        exercise2 = row['exercise2']
        summary = row['common_summary']
        if exercise1 not in exercise_summary_dic.keys():
            exercise_summary_dic[exercise1] = []
        exercise_summary_dic[exercise1].append(summary)
        if exercise2 not in exercise_summary_dic.keys():
            exercise_summary_dic[exercise2] = []
        exercise_summary_dic[exercise2].append(summary)
        summaries[summary_idx] = summary
        summary_idx += 1
        if exercise1 not in all_exercises:
            all_exercises.add(exercise1)
            exercises[idx] = exercise1
            idx += 1
        if exercise2 not in all_exercises:
            all_exercises.add(exercise2)
            exercises[idx] = exercise2
            idx += 1
    for exercise in exercise_summary_dic:
        summary_texts = exercise_summary_dic[exercise]
        exercise_id = get_id(exercises, exercise)
        summary_ids = []
        for summary in summary_texts:
            summary_id = get_id(summaries, summary)
            summary_ids.append(summary_id)
        exercise_to_summaries[exercise_id] = summary_ids
    return exercises, summaries, exercise_to_summaries

if __name__ == "__main__":
    path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    text_pair = get_pair_text(path, TOTAL_EXERCISE=50, LEVEL_1_NODE_INC_FACTOR=3)
    print(len(text_pair))
    text1, text2 = text_pair[0][0], text_pair[0][1]
    print(text_pair[0])

## python test_4_similarity_matrix.py