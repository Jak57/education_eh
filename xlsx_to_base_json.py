import pandas as pd
import json

def excel_to_json(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file)  # Assumes first row is header

    # Get the column containing prompts (Column A / "Result")
    prompts = df.iloc[:, 0].dropna().tolist()

    # Build the JSON structure
    data = {
        "levels": [
            {
                "level": 0,
                "nodes": [
                    {
                        "id": idx,
                        "prompt": p,
                        "match": None,
                        "summary": None,
                        "score": None,
                        "evaluation": None,
                        "alive": True
                    }
                    for idx, p in enumerate(prompts)
                ]
            }
        ]
    }

    # Write to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON written to {output_file}")


# Example usage:
# excel_to_json("UF_Code.xlsx", "UF_Code.json")


# ## Dissimilar (WB)
# input_xlsx_path = "FINAL_random_prompts_64.xlsx"
# output_json_path = "Results/Gemma/FINAL_random_prompts_64.json"

## Dissimilar (Nemotron-SFT-Science)
input_xlsx_path = "NT_Random_981.xlsx"
output_json_path = "Results/Gemma/NT_random_prompts_64.json"

excel_to_json(input_xlsx_path, output_json_path) # JKH