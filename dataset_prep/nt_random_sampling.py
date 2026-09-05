import random
import unicodedata
# random.seed(42)

import pandas as pd
# --------------------------------------------------------------------------- #
# Garbled / non-English character detection (same rules as
# extract_specificity_mturk_csv.py: keep ASCII, typography, Latin/Greek letters;
# flag other scripts, math-alphanumerics, combining marks, control/format chars)
# --------------------------------------------------------------------------- #
_ALLOWED_LETTER_PREFIXES = ("LATIN", "GREEK")

def _is_bad_char(c: str) -> bool:
    cp = ord(c)
    if cp < 128:
        return False
    if 0x1D400 <= cp <= 0x1D7FF:
        return True
    cat = unicodedata.category(c)
    if cat in ("Mn", "Mc", "Me"):
        return True
    if cat in ("Cc", "Cf", "Cs", "Co"):
        return True
    if cat[0] == "L":
        try:
            name = unicodedata.name(c)
        except ValueError:
            return True
        if not name.startswith(_ALLOWED_LETTER_PREFIXES):
            return True
    return False

def is_clean(s: str) -> bool:
    return not any(_is_bad_char(c) for c in s)

def clean_text(text):
    if ")." in text:
        idx = text.find(").")
        text = text[:idx+len(").")]
    if "}." in text:
            idx = text.find("}.")
            text = text[:idx+len("}.")]
    text_without_prefix = text.strip()
    if "A:" in text_without_prefix:
        idx = text_without_prefix.find("A:")
        return text_without_prefix[:idx]
    return text_without_prefix

def save_eligible_exercises(path, min_chars=50, max_chars=4000) -> list[str]:
    df = pd.read_csv(path)
    df['instructions'] = df['instructions'].apply(clean_text)
    texts = df["instructions"].dropna().astype(str)
    keep = texts[texts.str.len().between(min_chars, max_chars) & texts.map(is_clean)]
    pool = list(dict.fromkeys(keep))
    df_new = df.head(len(pool))
    df_new['instructions'] = pool
    df_new.to_csv(path, index=False)

def update_score_file(path):
    save_eligible_exercises(path)
    print(f"Updated file saved at {path}.")

def prepare_64_sample(score_filepath, xlsx_filepath):
    df = pd.read_csv(score_filepath)
    samples = []
    for index, row in df.iterrows():
        samples.append(row['instructions'])
    random.shuffle(samples)
    data = {
        'Result': samples[:64]
    }
    df1 = pd.DataFrame(data, columns=data.keys())
    df1.to_excel(xlsx_filepath, index=False)
    print(f"Random 64 samples saved at {xlsx_filepath}.")

if __name__ == "__main__":
    score_filepath = "dataset_NT_with_scores_so.csv"
    update_score_file(score_filepath) ## Uncomment for the first run, then comment out.

    idx = random.randint(1, 1000)
    xlsx_filepath = f"NT_Random_{idx}.xlsx"
    prepare_64_sample(score_filepath, xlsx_filepath)