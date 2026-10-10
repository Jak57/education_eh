from test_4_similarity_matrix import get_mapping
from test_6_entailment_link import get_id
from test_1_KC_gen import build_tree, build_dag
import json

from utils import load_csv

def get_exercise_text(text_with_sol):
    prefix = "\n\n##Here are"
    idx = text_with_sol.find(prefix)
    return text_with_sol[:idx]

def get_summary_to_exercise(exercise_to_summaries):
    summaries_to_exercise = {}
    for exercise_id in exercise_to_summaries:
        summary_ids = exercise_to_summaries[exercise_id]
        for summary_id in summary_ids:
            if summary_id not in summaries_to_exercise.keys():
                summaries_to_exercise[summary_id] = []
            summaries_to_exercise[summary_id].append(exercise_id)
    return summaries_to_exercise

def perform_deduplication(
        exercise_summary_map_path, 
        entail_score_path,
        tree_path_entail,
        dag_path_entail,
        TOTAL_EXERCISE,
        ENTAIL_THRESHOLD
    ):
    exercises, summaries, exercise_to_summaries = get_mapping(exercise_summary_map_path, TOTAL_EXERCISE)
    summaries_to_exercise = get_summary_to_exercise(exercise_to_summaries)
    df = load_csv(entail_score_path)
    for index, row in df.iterrows():
        if row['entail_score'] > ENTAIL_THRESHOLD:
            summary_id = get_id(summaries, row['summary'])
            problem_id = get_id(exercises, row['exercise'])
            summaries_to_exercise[summary_id].append(problem_id)
    summary_unique = []
    summary_exercise = []
    for key in summaries_to_exercise:
        exercise_sorted = sorted(summaries_to_exercise[key])
        if exercise_sorted not in summary_unique:
            summary_unique.append(exercise_sorted)
            summary_exercise.append((key, exercise_sorted))
    exercise_to_summaries_deduplicated = {}
    for item in summary_exercise:
        summary_id = item[0]
        exercise_ids = item[1]
        for exercise_id in exercise_ids:
            if exercise_id not in exercise_to_summaries_deduplicated:
                exercise_to_summaries_deduplicated[exercise_id] = []
            exercise_to_summaries_deduplicated[exercise_id].append(summary_id)
    summary_ids_dedup = set()
    exercise_kc_list = []
    for key in exercise_to_summaries_deduplicated:
        for summary_id in exercise_to_summaries_deduplicated[key]:
            summary_ids_dedup.add(summary_id)
            exercise_kc_list.append((exercises[key], summaries[summary_id]))
    kc_idx = TOTAL_EXERCISE
    summaries_updated = {}
    unique_kcs = set()
    for item in exercise_kc_list:
        kc = item[1]
        if kc not in unique_kcs:
            summaries_updated[kc_idx] = kc
            unique_kcs.add(kc)
            kc_idx += 1
    exercise_to_kc_updated = {}
    for item in exercise_kc_list:
        exercise = item[0]
        summary = item[1]
        exercise_id = get_id(exercises, exercise)
        summary_id = get_id(summaries_updated, summary)
        if exercise_id not in exercise_to_kc_updated.keys():
            exercise_to_kc_updated[exercise_id] = []
        exercise_to_kc_updated[exercise_id].append(summary_id)
    build_tree(exercises, summaries_updated, exercise_to_kc_updated, tree_path_entail)
    build_dag(tree_path_entail, dag_path_entail)
    return exercises, summaries_updated, exercise_to_kc_updated

def save_exercise_kc_mapping(
        exercise_summary_map_path, 
        entail_score_path, 
        tree_path_entail,
        dag_path_entail,
        exercise_kc_map_path,
        TOTAL_EXERCISE, 
        ENTAIL_THRESHOLD
    ):
    exercises, summaries, exercise_to_kc = perform_deduplication(
        exercise_summary_map_path, 
        entail_score_path, 
        tree_path_entail,
        dag_path_entail,
        TOTAL_EXERCISE, 
        ENTAIL_THRESHOLD
    )
    exercise_to_kc_dic = {}
    for problem_id in exercise_to_kc:
        summary_ids = exercise_to_kc[problem_id]
        problem_text_with_sol = exercises[problem_id]
        exercise = get_exercise_text(problem_text_with_sol)
        if exercise not in exercise_to_kc_dic.keys():
            exercise_to_kc_dic[exercise] = []
        for summary_id in summary_ids:
            summary = summaries[summary_id]
            exercise_to_kc_dic[exercise].append(summary)
    with open(exercise_kc_map_path, 'w') as f:
            json.dump(exercise_to_kc_dic, f)
    print(f"Exercise-to-KC dictionary saved at {exercise_kc_map_path}")

if __name__ == "__main__":
    exercise_summary_map_path = "outputs/text_summary_map_150.csv" 
    entail_score_path = "outputs/text_summary_entail.csv"
    tree_path_entail = "visualisation/test_tree_entail_dedup.json"
    dag_path_entail = "visualisation/test_dag_entail_dedup.json"
    exercise_kc_map_path = "outputs/problem_kc_5_CodeWorkout_claude.json"

    save_exercise_kc_mapping(
        exercise_summary_map_path, 
        entail_score_path, 
        tree_path_entail,
        dag_path_entail,
        exercise_kc_map_path,
        TOTAL_EXERCISE=50, 
        ENTAIL_THRESHOLD=0.7
    )

# python test_7_deduplication.py