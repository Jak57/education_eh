from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    message=[
        {
            "role": "user",
            "content": "What is photosynthesis? Explain in two sentences."
        }
    ],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)