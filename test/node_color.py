from utils import load_json, load_xlsx, write_to_json

def add_subject_information(path_xlsx, path_tree_json, path_tree_json_new):
    data_json = load_json(path_tree_json)
    data_xlsx = load_xlsx(path_xlsx)
    subjects = []
    for index, row in data_xlsx.iterrows():
        subjects.append(row['Subject'])
    new_json = []
    for idx, item in enumerate(data_json):
        if idx >= len(subjects):
            item['subset'] = "mixed"
        else:
            item['subset'] = subjects[idx].lower()
        new_json.append(item)
    write_to_json(path_tree_json_new, new_json)

if __name__ == "__main__":
    path_json = "visualisation/tree_data.json"
    path_xlsx = "NT_Random_758_with_solution1.xlsx"
    path_json_new = "visualisation/tree_data_with_subject.json"
    add_subject_information(path_xlsx, path_json, path_json_new)

