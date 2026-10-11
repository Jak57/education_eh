from utils import get_system_prompt_kc, get_kc_generation_prompt
from test_1_KC_gen import get_api_key
from utils import load_json
from test_0_generate_jina_embed import load_model, generate_embeddings

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
    new_dic = {}
    problem_cnt = 0
    for dic in data:
        kc = dic['kc']
        summary = dic['summary']
        if isinstance(kc, str):
            try:
                kc1 = ast.literal_eval(kc)
                if type(kc1) == type([]):
                    new_dic[summary] = kc1
                elif type(kc1) == type({}):
                    new_dic[summary] = list(kc1.values())
                else:
                    print("Problem: Not valid literal")
                    print(kc1)
                    new_dic[summary] = []
                    problem_cnt += 1
            except:
                new_dic[summary] = []
                problem_cnt += 1
                print("Problem: Not literal - ", type(kc), kc)
                pass
    print(f"Total problematic extraction: {problem_cnt}")
    return new_dic

def get_correct_mapping(exercise_kc_map_path, generated_kc_path):
    correct_map = kc_sanity_check(generated_kc_path)
    new_dic = {}
    data = load_json(exercise_kc_map_path)
    for exercise in data:
        summaries = data[exercise]
        kcs = []
        total_split_count = 0
        for summary in summaries:
            kc_split = correct_map[summary]
            total_split_count += len(kc_split)
            for kc in kc_split:
                if type(kc) == str:
                    kc = kc.lower()
                    if kc not in kcs:
                        kcs.append(kc)
        new_dic[exercise] = sorted(kcs)
    return new_dic

def save_kc_embedding(exerecise_kc_map, embedding_path):
    kc_set = set()
    for key in exerecise_kc_map:
        kcs = exerecise_kc_map[key]
        for kc in kcs:
            kc_set.add(kc)
    kc_set = sorted(list(kc_set))
    print(f"\nTotal kc = {len(kc_set)}")
    model = load_model()
    corpus_sentences, corpus_embeddings = generate_embeddings(
        model,
        kc_set,
        embedding_path,
    )

if __name__ == "__main__":
    input_path, output_path = "outputs/problem_kc_5_CodeWorkout_claude.json", "outputs/problem_kc_5_CodeWorkout_claude_kc_split.json"
    # final_exercise_kc_map_path = "outputs/problem_kc_5_CodeWorkout_claude_143_kcs.json"
    kc_embedding_path = "outputs/embedding/stage3/kc_embed.pkl"

    api_keys = get_api_key()

    ## Uncomment this to generate KC splits from common summaries
    # generate_kc_mapping(input_path, output_path, api_keys)

    map_with_duplicate = get_correct_mapping(input_path, output_path)
    save_kc_embedding(map_with_duplicate, kc_embedding_path)


## python test_9_split_common_skill.py