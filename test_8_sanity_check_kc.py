import json

from utils import load_json

if __name__ == "__main__":
    comprehend_path = "outputs/problem_kc_5_CodeWorkout_claude.json"
    kcgen_path = "dataset/problem_kc_5_CodeWorkout.json"
    data = load_json(kcgen_path)

    data_comp = load_json(comprehend_path)

    # print(data.keys())
    exercises = []
    for key in data:
        exercises.append(key)

    exercise_comp = []
    for key in data_comp:
        exercise_comp.append(key)

    if sorted(exercises) == sorted(exercise_comp):
        print("yes")
    else:
        print("no")

    data = data_comp
    print(type(data))
    for idx, key in enumerate(data):
        print(idx+1)
        print(len(data[key]), type(data[key]))
        print(key)
        for i, item in enumerate(data[key]):
            print(f"--------------------> {i+1}:", item)
        print()

## python test_8_sanity_check_kc.py