# import os
# import csv
# import pickle
# import time
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# import sys
# import traceback
# from tqdm import tqdm

# dataset_path = "./10NN_L3.csv"
# output_csv_path = 'outputs/10NN_L3_search_results.csv'
# max_corpus_size = 6

# def load_model(model_name='all-mpnet-base-v2'):
#     print(f"Loading SentenceTransformer model: {model_name}")
#     return SentenceTransformer(model_name)

# def generate_embeddings(model, corpus_sentences, embedding_cache_path):
#     print("Embedding cache not found. Creating new embeddings...")
#     print("Encoding the corpus. This might take a while...")

#     corpus_embeddings = []
#     for sentence in tqdm(corpus_sentences, desc = 'Encoding Sentences:'):
#         corpus_embeddings.append(model.encode(sentence, show_progress_bar=True, convert_to_numpy=True))
#     corpus_embeddings = np.array(corpus_embeddings)
#     corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
#     print("Storing embeddings on disk...")
#     os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
#     with open(embedding_cache_path, "wb") as fOut:
#         pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    
#     print("Embeddings stored successfully")
#     return corpus_sentences, corpus_embeddings

# def load_embeddings(embedding_cache_path):
#     print("Loading precomputed embeddings...")
#     with open(embedding_cache_path, "rb") as fIn:
#         cache_data = pickle.load(fIn)
#     print("Embeddings loaded successfully")
#     corpus_sentences = cache_data['sentences']
#     corpus_embeddings = cache_data['embeddings']
#     print("Pre-computed embeddings loaded successfully")
#     return corpus_sentences, corpus_embeddings 

# def create_faiss_index(embeddings, embedding_size=768):
#     """Creates and populates a FAISS index for nearest neighbor search."""
#     n_clusters = 2

#     print("Creating FAISS index...")
#     quantizer = faiss.IndexFlatIP(embedding_size)
#     index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
#     index.add(embeddings)
#     index.nprobe = 3
#     print("FAISS index populated")
#     return index

# def find_nearest_neighbors(corpus_sentences, model_name='all-mpnet-base-v2', top_k=2):
#     if len(corpus_sentences) < 2:
#         return []
    
#     model = load_model(model_name)

#     embedding_cache_path = f'outputs/{model_name.replace("/", "_")}_embeddings.pkl'    
#     if not os.path.exists(embedding_cache_path):
#         corpus_sentences, corpus_embeddings = generate_embeddings(model, corpus_sentences, embedding_cache_path)
#     else:
#         corpus_sentences, corpus_embeddings = load_embeddings(embedding_cache_path)
    
#     index = create_faiss_index(corpus_embeddings, corpus_embeddings.shape[1])
    
#     nearest_neighbors = []
#     used_indices = set()  # To track used sentences and avoid duplicate pairing

#     print("Finding nearest neighbors...")
#     for i in tqdm(range(len(corpus_sentences)), desc="Processing pairs"):
#         if i in used_indices:
#             continue  # Skip if already paired

#         question_embedding = np.expand_dims(corpus_embeddings[i], axis=0)
#         distances, corpus_ids = index.search(question_embedding, top_k)

#         for j in range(1, top_k):  # Skip self-match (index 0 is always the query itself)
#             nn_id = corpus_ids[0][j]
#             if nn_id in used_indices:
#                 continue  # Skip if already used

#             nearest_neighbors.append((corpus_sentences[i], corpus_sentences[nn_id]))
#             used_indices.add(i)
#             used_indices.add(nn_id)
#             break  # Move to the next sentence

#     print(f"Found {len(nearest_neighbors)} nearest neighbor pairs")
#     return nearest_neighbors

# def read_stories(file_path, max_size):
#     print(f"Reading stories from {file_path}...")
#     corpus_sentences = []
#     with open(file_path, 'r', encoding='utf8') as f:
#         csv_reader = csv.reader(f)
#         next(csv_reader)
#         for row in csv_reader:
#             if row:
#                 instruction = row[0].strip()  
#                 corpus_sentences.append(instruction)
#                 if len(corpus_sentences) >= max_size:
#                     break
#     print(f"Read {len(corpus_sentences)} unique entries")
#     return list(zip(corpus_sentences))

# if __name__ == "__main__":
#     print("Starting the retrieval")
#     corpus_data = read_stories(dataset_path, max_corpus_size)
#     corpus_sentences = [data[0] for data in corpus_data]




import os
import pickle
import faiss
import numpy as np
import openpyxl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from tree import Tree

def load_model(model_name='all-mpnet-base-v2'):
    print(f"Loading SentenceTransformer model: {model_name}")
    return SentenceTransformer(model_name)

def generate_embeddings(model, corpus_sentences, embedding_cache_path):
    print("Encoding the corpus. This might take a while...")
    # corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
    corpus_embeddings = []
    for sentence in tqdm(corpus_sentences, desc="Encoding sentences"):
        corpus_embeddings.append(model.encode(sentence, show_progress_bar=True, convert_to_numpy=True))
    corpus_embeddings = np.array(corpus_embeddings)
    corpus_embeddings /= np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    
    print("Storing embeddings on disk...")
    os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)  
    print("Embeddings stored successfully")
    return corpus_sentences, corpus_embeddings

def load_embeddings(embedding_cache_path):
    print("Loading precomputed embeddings...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
    print("Embeddings loaded successfully")
    return cache_data['sentences'], cache_data['embeddings']

def create_faiss_index(embeddings, embedding_size=768, n_clusters=2):
    print("Creating FAISS index...")
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    print("Adding embeddings to the index...")
    for i in tqdm(range(0, len(embeddings), 1000), desc="Adding to FAISS index"):
        index.add(embeddings[i:i+1000])
    # index.add(embeddings)
    index.nprobe = 3
    print("FAISS index populated")
    return index


def find_third(tree:Tree, level_idx:int, model_name='all-mpnet-base-v2'):
    model = load_model(model_name)
    level = tree.levels[level_idx]        # ← the Level object we’ll operate on

    active_idxs = []                      # ← node indexes, not sheet rows
    corpus_sentences = []

    print("\nProcessing rows:")
    for idx, node in enumerate(level.nodes):
        if not node.alive:
            continue 
        val = node.prompt
        # print(f"Idx {idx}, Value: {val}")
        if val:
            if str(val).startswith("Error:"):
                continue
                # print(f"Skipping row {row} due to Error: prefix")
            else:
                active_idxs.append(idx)
                corpus_sentences.append(val)
                # print(f"Added row {row} to active_rows")

    print(f"\nTotal active rows: {len(active_idxs)}")
    print(f"Active row numbers: {active_idxs}")
    
    embedding_cache_path = f'outputs/{model_name.replace("/", "_")}_embeddings.pkl'

    sentences, embeddings = generate_embeddings(model, corpus_sentences, embedding_cache_path)
    
    for idx in active_idxs:
        node = level.nodes[idx]
        query = node.prompt
        third, score = nearest_neighbor_third(query, sentences, embeddings)
        node.third_score = score

        if third is None:
            node.third = None
            continue

        for idx2 in active_idxs:
            if level.nodes[idx2].prompt == third:
                node.third = idx2
                break
    # tree.dump("10NNofASpecificity.json")

        
def nearest_neighbor_third(query, sentences, embeddings):
    query_embedding = embeddings[sentences.index(query)]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    embedding_size = 768
    top_k_hits = 3
    n_clusters = 2

    num_sentences = len(sentences)

    # If we don't have enough points for [self, nearest, second-nearest], bail out.
    if num_sentences < 3:
        return None, 0.0

    # Normalize embeddings for inner-product = cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Never ask for more hits than the number of points
    top_k_hits = min(top_k_hits, num_sentences)

    # If too few points to train IVF, fall back to a flat index (no training).
    if num_sentences < n_clusters:
        index = faiss.IndexFlatIP(embedding_size)
        index.add(embeddings)
    else:
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 3
        index.train(embeddings)
        index.add(embeddings)

    distances, corpus_ids = index.search(query_embedding, top_k_hits)

    hits = [{'corpus_id': int(i), 'score': float(s)} for i, s in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)

    # Need 3 hits to take hits[2]
    if len(hits) < 3:
        return None, 0.0

    third = sentences[hits[2]['corpus_id']]
    score = hits[2]['score']
    return third, score

    

