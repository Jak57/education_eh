from test_4_similarity_matrix import get_mapping
from test_0_generate_jina_embed import load_model, generate_embeddings, get_cosine_similarity

import heapq

# def get_kc_id(summaries):
#     return list(summaries.keys())

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

def add_entailment_links(exercise_summary_map_path, exercise_embedding_path, summary_embedding_path, TOTAL_EXERCISE, TOP_SIM=5):
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
            # print(problem_idx, skill_idx, similarity)
            heapq.heappush(hq, (-similarity, (problem_id, skill_id)))
            # break
        # break

        # Extract top TOP_SIM nodes
        cnt = TOP_SIM
        while len(hq) > 0 and cnt > 0:
            top = heapq.heappop(hq)
            # print(top)
            _, index = top
            i, j = index
            final_candidate[problem_id].append(j)
            cnt -= 1

        # print("\n\n")
        # break
    for key in final_candidate:
        print(key, len(final_candidate[key]))
        print(final_candidate[key])
        print()




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
    # save_summary_embedding(output_embedding_path, TOTAL_EXERCISE=50)

    add_entailment_links(exercise_summary_map_path, exercise_embedding_path, summary_embedding_path, TOTAL_EXERCISE=50)

    # exercise_text, skill_text, similarity_matrix = get_cosine_similarity(exercise_embedding_path, output_embedding_path)
    # for i in range(10):
    #     for j in range(10):
    #         print(f"{similarity_matrix[i][j]:.2f}", end=" ")
    #     print()
    

## python test_6_entailment_link.py

