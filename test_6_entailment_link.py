from test_4_similarity_matrix import get_mapping
from test_0_generate_jina_embed import load_model, generate_embeddings, get_cosine_similarity
from evaluation.simple_eval import get_entailment_score
from test_1_KC_gen import get_api_key
from test_1_KC_gen import build_tree, build_dag

import heapq
import pandas as pd
import copy

def get_id(dic, text):
    for key in dic:
        if dic[key] == text:
            return key
    return None

def is_id_present(dic, exercise_id, skill_id):
    for key in dic:
        if key == exercise_id:
            if skill_id in dic[key]:
                return True
    return False

def get_index(items, text):
    for i in range(len(items)):
        if items[i] == text:
            return i
    return None

def add_entailment_links(
        exercise_summary_map_path, 
        exercise_embedding_path, 
        summary_embedding_path, 
        entail_score_path, 
        tree_path_entail,
        dag_path_entail,
        TOTAL_EXERCISE=50, 
        TOP_SIM=5,
        ENTAIL_THRESHOLD=0.7,
    ):
    exercises, summaries, exercise_to_summaries = get_mapping(exercise_summary_map_path, TOTAL_EXERCISE)
    exercise_texts, skill_texts, similarity_matrix = get_cosine_similarity(exercise_embedding_path, summary_embedding_path)
    candidate_skill_id = {}
    for i in range(len(exercise_texts)):
        exercise_id = get_id(exercises, exercise_texts[i])
        candidate_skill_id[exercise_id] = []
        for j in range(len(skill_texts)):
            skill_id = get_id(summaries, skill_texts[j])
            if not is_id_present(exercise_to_summaries, exercise_id, skill_id):
                candidate_skill_id[exercise_id].append(skill_id)
    final_candidate = {}
    for problem_id in candidate_skill_id:
        skill_ids = candidate_skill_id[problem_id]
        problem_text = exercises[problem_id]
        problem_idx = get_index(exercise_texts, problem_text)
        hq = []
        final_candidate[problem_id] = []
        for skill_id in skill_ids:
            skill_text = summaries[skill_id]
            skill_idx = get_index(skill_texts, skill_text)
            similarity = similarity_matrix[problem_idx][skill_idx]
            heapq.heappush(hq, (-similarity, (problem_id, skill_id)))
        cnt = TOP_SIM
        while len(hq) > 0 and cnt > 0:
            top = heapq.heappop(hq)
            _, index = top
            i, j = index
            final_candidate[problem_id].append(j)
            cnt -= 1
    exercise_to_summaries_copy = copy.deepcopy(exercise_to_summaries)
    api_keys = get_api_key()
    final_exercise_list = []
    final_summary_list = []
    final_entail_list = []
    for key in final_candidate:
        exercise = exercises[key]
        for summary_id in final_candidate[key]:
            summary = summaries[summary_id]
            entail_score = get_entailment_score(exercise, summary, api_keys=api_keys)
            if entail_score >= ENTAIL_THRESHOLD:
                exercise_to_summaries_copy[key].append(summary_id)
            final_exercise_list.append(exercises[key])
            final_summary_list.append(summaries[summary_id])
            final_entail_list.append(entail_score)
    data = {
        'exercise': final_exercise_list,
        'summary': final_summary_list,
        'entail_score': final_entail_list
    }
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(entail_score_path, index=False)
    print(f"Entailment scores saved at {entail_score_path}")
    build_tree(exercises, summaries, exercise_to_summaries_copy, tree_path_entail)
    build_dag(tree_path_entail, dag_path_entail)
    return exercises, summaries, exercise_to_summaries_copy

def save_summary_embedding(output_path, TOTAL_EXERCISE):
    exercises, summaries, exercise_to_summaries = get_mapping(exercise_summary_map_path, TOTAL_EXERCISE)
    summary_texts = list(summaries.values())
    model = load_model()
    print(f"Generating embedding for {len(summary_texts)} common summaries...")
    embed = generate_embeddings(model, summary_texts, output_path)
    print(f"Embedding saved at {output_path}")

if __name__ == "__main__":
    exercise_summary_map_path = "outputs/text_summary_map_150.csv"
    exercise_embedding_path = "outputs/jinaai_jina-code-embeddings-1.5b_embeddings.pkl"
    summary_embedding_path = "outputs/jina_embed_summary_path.pkl"
    entail_score_path = "outputs/text_summary_entail.csv"

    tree_path_entail = "visualisation/test_tree_entail.json"
    dag_path_entail = "visualisation/test_dag_entail.json"

    # save_summary_embedding(output_embedding_path, TOTAL_EXERCISE=50)

    exercises, summaries, exercise_to_summaries_entail = add_entailment_links(
        exercise_summary_map_path, 
        exercise_embedding_path, 
        summary_embedding_path, 
        entail_score_path, 
        tree_path_entail,
        dag_path_entail,
        TOTAL_EXERCISE=50, 
        TOP_SIM=5, 
        ENTAIL_THRESHOLD=0.7
    )

## python test_6_entailment_link.py

