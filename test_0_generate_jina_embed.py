import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from utils import load_xlsx

# ---- Embedding device / VRAM knobs ----
USE_CUDA = False  # set True to use GPU if available
EMBED_BATCH_SIZE = 8 #64  # lower this (e.g., 16/32) if you hit CUDA OOM; raise on CPU

def load_model(
        model_name='jinaai/jina-code-embeddings-1.5b',
        *,
        use_cuda: bool = False
):
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(
        f"Loading model: {model_name} "
        f"on device={device} "
        f"(USE_CUDA={use_cuda}, "
        f"cuda_available={torch.cuda.is_available()})"
    )
    if device == "cuda":
        model = SentenceTransformer(
            model_name,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
            tokenizer_kwargs={
                "padding_side": "left",
            },
            device=device,
        )
    else:
        model = SentenceTransformer(
            model_name,
            device=device,
        )
    return model

def generate_embeddings(
    model,
    corpus_sentences,
    embedding_cache_path,
    prompt_name='code2code_document',
):
    print("Encoding the corpus...")
    corpus_embeddings = model.encode(
        corpus_sentences,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        prompt_name=prompt_name
    )
    # Normalize the cosine similarity
    denom = np.linalg.norm(
        corpus_embeddings,
        axis=1,
        keepdims=True
    )
    denom[denom == 0] = 1.0
    corpus_embeddings = corpus_embeddings / denom
    print("Embedding shape:", corpus_embeddings.shape)
    print("Storing embeddings on disk...")
    os.makedirs(
        os.path.dirname(embedding_cache_path),
        exist_ok=True
    )
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump(
            {
                'sentences': corpus_sentences,
                'embeddings': corpus_embeddings
            },
            fOut
        )
    print("Embeddings stored successfully")
    return corpus_sentences, corpus_embeddings

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully")
    return cache_data['sentences'], cache_data['embeddings']

if __name__ == "__main__":
    # print("hello world")

    text = []

    df = load_xlsx("CW_Random_with_solution.xlsx")
    for index, row in df.iterrows():
        text.append(row['Result'])

    # print(len(text))
    text = text  #[:10]

    model = load_model()
    embed = generate_embeddings(model, text, "outputs/jina_embed_path.pkl")

## python test_0_generate_jina_embed.py