from utils import get_system_prompt_kc, get_kc_generation_prompt
from test_1_KC_gen import get_api_key
from utils import load_json

import json
import anthropic
import re
from openai import OpenAI 
import ast

def get_kc(summary,  model_name='claude', api_keys=None):
    system_prompt = get_system_prompt_kc()
    prompt14 = get_kc_generation_prompt(summary)
    if model_name == "claude":
            client = anthropic.Anthropic(
                api_key=api_keys["claude"]
            )
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = message.content[0].text.strip()
            match = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
            summary = match.group(1).strip() if match else raw.strip()
            return summary
    elif model_name == "openai":
        client = OpenAI(
            api_key=api_keys["openai"]
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt14}
            ]
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
        summary = match.group(1).strip() if match else raw.strip()
        return summary
    return []

def get_unique_common_summaries(data):
    common_summary_set = set()
    for problem in data:
        common_summaries = data[problem]
        for summary in common_summaries:
             common_summary_set.add(summary)
    return list(common_summary_set)

def generate_kc_mapping(input_path, output_path, api_keys):
    data = load_json(input_path)
    unique_summaries = get_unique_common_summaries(data)
    kc_list = []
    for idx, summary in enumerate(unique_summaries):
        kcs = get_kc(summary, model_name="openai", api_keys=api_keys)
        dic = {}
        dic['summary'] = summary
        dic['kc'] = kcs
        kc_list.append(dic)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kc_list, f, ensure_ascii=False, indent=4)
    print(f"File saved at {output_path}")

def kc_sanity_check(path):
    data = load_json(path)
    cnt = 1
    new_dic = {}
    cnt = 0
    problem_cnt = 0
    for dic in data:
        kc = dic['kc']
        summary = dic['summary']
        if isinstance(kc, str):
            cnt += 1
            try:
                kc1 = ast.literal_eval(kc)
                new_dic[summary] = kc1
            except:
                new_dic[summary] = []
                problem_cnt += 1
                print("Problem:")
                print(summary)
                print(type(kc), kc)
                print("\n\n")
                pass
    # print(cnt, cnt1)
    print(f"Total problematic format: {problem_cnt}")

    # with open("dataset/exercise_kc_map_claude.json", 'w') as f:
    #     json.dump(new_dic, f)
         

if __name__ == "__main__":
    input_path, output_path = "outputs/problem_kc_5_CodeWorkout_claude.json", "outputs/problem_kc_5_CodeWorkout_claude_kc_split.json"
    api_keys = get_api_key()

    ## Uncomment this to generate KC splits from common summaries
    # generate_kc_mapping(input_path, output_path, api_keys)

    kc_sanity_check(output_path)



## python test_9_split_common_skill.py