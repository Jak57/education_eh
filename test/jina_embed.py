# !pip install transformers>=4.53.0 torch>=2.7.1

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

INSTRUCTION_CONFIG = {
    "nl2code": {
        "query": "Find the most relevant code snippet given the following query:\n",
        "passage": "Candidate code snippet:\n"
    },
    "qa": {
        "query": "Find the most relevant answer given the following question:\n",
        "passage": "Candidate answer:\n"
    },
    "code2code": {
        "query": "Find an equivalent code snippet given the following code snippet:\n",
        "passage": "Candidate code snippet:\n"
    },
    "code2nl": {
        "query": "Find the most relevant comment given the following code snippet:\n",
        "passage": "Candidate comment:\n"
    },
    "code2completion": {
        "query": "Find the most relevant completion given the following start of code snippet:\n",
        "passage": "Candidate completion:\n"
    }
}

MAX_LENGTH = 8192

def cosine_similarity(x,y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return x @ y.T

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def add_instruction(instruction, query):
    return f'{instruction}{query}'

# The queries and documents to embed
queries = [
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["query"], "print hello world in python"),
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["query"], "initialize array of 5 zeros in c++")
]
documents = [
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["passage"], "print('Hello World!')"),
    add_instruction(INSTRUCTION_CONFIG["nl2code"]["passage"], "int arr[5] = {0, 0, 0, 0, 0};")
]
all_inputs = queries + documents

tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-code-embeddings-1.5b')
model = AutoModel.from_pretrained('jinaai/jina-code-embeddings-1.5b')

batch_dict = tokenizer(
    all_inputs,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
batch_dict.to(model.device)
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
query_embeddings = embeddings[:2]
passage_embeddings = embeddings[2:]

# Compute the (cosine) similarity between the query and document embeddings
scores = cosine_similarity(query_embeddings, passage_embeddings)
print(scores)
# tensor([[0.7647, 0.1115],
#         [0.0930, 0.6606]], grad_fn=<MmBackward0>)
