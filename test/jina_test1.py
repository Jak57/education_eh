import torch
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer

EMBED_BATCH_SIZE = 8

def load_model(
        model_name='jinaai/jina-code-embeddings-1.5b',
        *,
        use_cuda: bool = False
):
    pass
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

def get_nearest_neighbors_blossom(
        # tree: Tree,
        level_idx: int,
        model_name: str = 'jinaai/jina-code-embeddings-1.5b',
        *,
        use_custom_score: bool = True,
        # custom_threshold: Optional[float] = None,
        llm_model_name: str = 'openai',
        api_keys: dict = None,
):
    pass

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
    # pass


model = load_model()
corpus_sentences = ["""public int caughtSpeeding(int speed, boolean isBirthday)
{
    if (isBirthday)
    {
        speed = speed - 5;
    }
    if (speed <= 60)
    {
        return 0;
    }
    else if (speed <=80)
    {
        return 1;
    }
    else 
    {
        return 2;
    }
}
""", """public String alarmClock(int day, boolean vacation)
{
    if (vacation == true)
    {
        if (day == 0)
        {
        	return "off";
        }
        else if (day == 6)
        {
        	return "off";
        }
        else
        {
            return "10:00";
        }
    }
    else
    {
    	if (day == 0)
        {
        	return "10:00";
        }
        else if (day == 6)
        {
        	return "10:00";
        }
        else
        {
            return "7:00";
        }
    }
}
"""]
embedding_cache_path = "out/text.pkl"
corpus_sentences, corpus_embeddings = generate_embeddings(
    model,
    corpus_sentences,
    embedding_cache_path,
)