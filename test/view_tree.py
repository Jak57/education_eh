from utils import load_json
import pandas as pd

def get_exercise_kc_map(path, LEVEL_ID=0):
    data = load_json(path)['levels']
    exercise_kc_map = {}
    for dic in data:
        level_id = dic['level']
        if level_id == LEVEL_ID:
            for item in dic['nodes']:
                exercise = item['prompt']
                kc = item['summary']
                if exercise not in exercise_kc_map.keys():
                    exercise_kc_map[exercise] = ""
                exercise_kc_map[exercise] = kc
    return exercise_kc_map

def get_unique_kc(path):
    data = load_json(path)
    for idx, key in enumerate(data):
        print(idx+1)
        print(len(data[key]), type(data[key]))
        print(key)
        for i, item in enumerate(data[key]):
            print(f"--------------------> {i+1}:", item)
        print()

def save_exercise_summary_kc_map(path, bloosm_tree_path):
    # bloosm_tree_path = "log/code_workout/jina_prompt_update1/CW_random_prompts_50_blossom.json"
    # analyze_tree_data(bloosm_tree_path)
    # path = "dataset/problem_summary.csv"
    exercise_kc_map = get_exercise_kc_map(bloosm_tree_path, LEVEL_ID=0)
    print(len(exercise_kc_map))
    problems = []
    summaries = []
    for key in exercise_kc_map:
        # print(len(exercise_kc_map[key]))
        # print(key)
        # print("---")
        # print(exercise_kc_map[key])
        # print("---------------------\n\n")
        problems.append(key)
        summaries.append(exercise_kc_map[key])
    data = {
        "exercise": problems,
        "summary": summaries
    }   
    df = pd.DataFrame(data, columns=data.keys())
    # print(df.head(1))
    df.to_csv(path, index=False)

def analyze_tree_data(bloosm_tree_path):
    pass

if __name__ == "__main__":
    bloosm_tree_path = "log/code_workout/jina_prompt_update1/CW_random_prompts_50_blossom.json"
    analyze_tree_data(bloosm_tree_path)

    ## Prepare Exercise-Summary-KC map
    path = "dataset/problem_summary.csv"
    # save_exercise_summary_kc_map(path, bloosm_tree_path)





