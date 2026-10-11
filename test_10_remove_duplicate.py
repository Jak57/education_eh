from test_0_generate_jina_embed import load_embeddings
from test_9_split_common_skill import get_correct_mapping

from pathlib import Path
import json

def process_kc_set(similarity_matrix, corpus_sentences, kcs, SIM_THRES=0.75):
    ids = []
    for kc in kcs:
        for idx, text in enumerate(corpus_sentences):
            if text == kc:
                ids.append(idx)
    state = [False] * len(ids)
    i = 0
    deduplicated_kcs = []
    while (i < len(ids) - 1):
        if not state[i]:
            state[i] = True
            j = i + 1
            idx_i = ids[i]
            deduplicated_kcs.append(corpus_sentences[idx_i])
            while j < len(ids):
                idx_j = ids[j]
                if (not state[j]) and (similarity_matrix[idx_i][idx_j] > SIM_THRES):
                    state[j] = True
                j += 1
        i += 1
    return deduplicated_kcs

def remove_duplicate_kc(exercise_summary_path, summary_kc_path, kc_embedding_path, deduplicated_exercise_kc_path, SIM_THRES):
    map_with_duplicate = get_correct_mapping(exercise_summary_path, summary_kc_path)
    corpus_sentences, corpus_embeddings = load_embeddings(kc_embedding_path)
    print(f"Total kc before deduplication: {len(corpus_sentences)}")
    kc_embed_map = {}
    for i in range(len(corpus_sentences)):
        kc_embed_map[corpus_sentences[i]] = corpus_embeddings[i]
    similarity_matrix = corpus_embeddings @ corpus_embeddings.T
    final_unique_set = set()
    map_without_duplication = {}
    for idx, key in enumerate(map_with_duplicate):
        deduplicated_kc = process_kc_set(similarity_matrix, corpus_sentences, map_with_duplicate[key], SIM_THRES)
        map_without_duplication[key] = deduplicated_kc
        for item in deduplicated_kc:
            final_unique_set.add(item)
    print(f"Total kc after deduplication {len(final_unique_set)}")
    path = Path(deduplicated_exercise_kc_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(map_without_duplication, f)
    print(f"File saved at {path}")

if __name__ == "__main__":
    input_path, output_path = "outputs/problem_kc_5_CodeWorkout_claude.json", "outputs/problem_kc_5_CodeWorkout_claude_kc_split.json"
    kc_embedding_path = "outputs/embedding/stage3/kc_embed.pkl"
    deduplicated_exercise_kc_path = "outputs/exercise_kc_map/150/problem_kc_5_CodeWorkout_claude_v2.json"
    remove_duplicate_kc(input_path, output_path, kc_embedding_path, deduplicated_exercise_kc_path, SIM_THRES=0.75)

## python test_10_remove_duplicate.py