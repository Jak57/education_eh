from utils import load_json
import ast
import json

if __name__ == "__main__":
    # pass
    path = "dataset/problem_summary_kc_claude.json"
    # path = "dataset/problem_summary_kc.json"

    kcgen_file = "dataset/problem_kc_5_CodeWorkout.json"

    data = load_json(path)

    problems = []
    data1 = load_json(kcgen_file)
    for prob in data1:
        # print(prob)
        problems.append(prob.strip())

    # print(data[0])
    cnt = 1
    new_dic = {}
    for dic in data:
        kc = dic['kc']
        # print(type(kc))

        exercise = dic['exercise']
        ori_ex = ""
        for item in problems:
                if item in exercise:
                    ori_ex = item
                    print("Yes", cnt)
                    cnt += 1

        if isinstance(kc, str):
            try:
                kc = ast.literal_eval(kc)
                print(ori_ex)
                print(kc)
                print("-----")
                new_dic[ori_ex] = kc
            except:
                print("problem--------------------->")
                print(dic['exercise'])
                print(type(kc), kc)
                pass
        # exercise = dic['exercise']

    with open("dataset/exercise_kc_map_claude.json", 'w') as f:
         json.dump(new_dic, f)
        # cnt = 1
        # for item in problems:
        #     if item in exercise:
        #         print("Yes", cnt)
        #         cnt += 1
                # break


        # if type(kc) != type([]):
        #     print("yes")
        #     print(kc)
        #     print()

        # problems = []
        # data1 = load_json(kcgen_file)
        # for prob in data1:
        #     # print(prob)
        #     problems.append(prob.strip())
            # break

        