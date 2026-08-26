"""Text-generation dispatch for the EvalTree pipeline.

Mirrors the provider dispatch used by the blossom modules
(common_summary_generation/summarize.py, evaluation/simple_eval.py):
one function, model_name in {"openai", "claude", "gemma", "gemini"},
credentials from the api_keys dict built in main.py.
"""

import anthropic
from openai import OpenAI


def generate_text(user_prompt: str, system_prompt: str,
                  model_name: str = "claude", api_keys: dict = None) -> str:
    if model_name == "openai":
        client = OpenAI(api_key=api_keys["openai"])
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    elif model_name == "claude":
        client = anthropic.Anthropic(api_key=api_keys["claude"])
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text.strip()
    elif model_name == "gemma":
        client = OpenAI(api_key=api_keys["gemma"]["api_key"],
                        base_url=api_keys["gemma"]["base_url"])
        resp = client.chat.completions.create(
            model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
            max_tokens=1024, stop=["<end_of_turn>"],
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    elif model_name == "gemini":
        client = OpenAI(api_key=api_keys["gemini"]["api_key"],
                        base_url=api_keys["gemini"]["base_url"])
        resp = client.chat.completions.create(
            model="gemini-2.0-flash", max_tokens=1024,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        return resp.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
