from utils import load_csv, load_xlsx
import pandas as pd

def get_valid_student_ids(path, total_problems=50, first_submission=True, student_visualisation_count=22):
    df = load_csv(path)
    students = {}
    for index, row in df.iterrows():
        student = row['SubjectID']
        if student not in students.keys():
            students[student] = set()
        students[student].add(row['ProblemID'])
    valid_ids = []
    for key in students:
        if len(students[key]) == total_problems:
            valid_ids.append(key)
    dic = {}
    for index, row in df.iterrows():
        student = row['SubjectID']
        if student in valid_ids:
            if student not in dic.keys():
                dic[student] = []
            dic[student].append(row)
    candidate_rows = []
    for student in dic:
        student_info = dic[student]
        problem_info_row = {}
        for info in student_info:
            problem_id = info['ProblemID']
            if problem_id not in problem_info_row.keys():
                problem_info_row[problem_id] = []
            problem_info_row[problem_id].append(info)
        for problem in problem_info_row:
            df1 = pd.DataFrame(problem_info_row[problem])
            # df1['ServerTimestamp'] = pd.to_datetime(df1['ServerTimestamp'])
            df1 = df1.sort_values("ServerTimestamp")
            if first_submission:
                candidate_rows.append(df1.iloc[0])
            else:
                candidate_rows.append(df1.iloc[-1])
    df3 = pd.DataFrame(candidate_rows)
    print(f"Total student that solved {total_problems} problems: {len(valid_ids)}")
    valid_ids = sorted(valid_ids)
    return valid_ids[:student_visualisation_count], df3

def get_problem_id_map(path_exercise_with_id):
    df = load_xlsx(path_exercise_with_id)
    dic = {}
    problem_ids = []
    for index, row in df.iterrows():
        problem_id = row['ProblemID']
        dic[str(problem_id)] = row['Result']
        problem_ids.append(int(problem_id))
    problem_ids = sorted(problem_ids)
    return problem_ids, dic

if __name__ == "__main__":
    total_problems=50
    path = "dataset/dataset_clean.csv"
    path_exercise_with_id = "CW_Random_with_solution.xlsx"
    output_path = "dataset_CW_with_scores.csv"

    valid_ids, df_valid = get_valid_student_ids(path, first_submission=True, student_visualisation_count=40)
    print(f"Total student selected for visualisation: {len(valid_ids)}")
    problem_ids, problem_dic = get_problem_id_map(path_exercise_with_id)
    data = {
        'id': problem_ids
    }
    df4 = pd.DataFrame(data, columns=data.keys())
    df4['source'] = ['CodeWorkout'] * total_problems
    exercises = []
    for problem_id in problem_ids:
        exercises.append(problem_dic[str(problem_id)])
    df4['instructions'] = exercises

    # Populate candidate student's data
    for index, student_id in enumerate(valid_ids):
        short_id = "student_" + str(index+1)
        id_score = []
        cnt = 0
        for idx, row in df_valid.iterrows():
            if student_id == row['SubjectID'].strip():
                cnt += 1
                id_score.append((row['ProblemID'], row['Score_x']))
        id_score = sorted(id_score)
        scores = []
        for item in id_score:
            scores.append(item[1])
        df4[short_id] = scores
    df4.to_csv(output_path, index=False)
    print(f"\nFile saved at {output_path}")