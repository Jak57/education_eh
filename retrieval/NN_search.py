import os
import csv
import pickle
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import sys
import traceback
from tqdm import tqdm


print("Starting the script...")
model_name = 'all-mpnet-base-v2'
print(f"Loading SentenceTransformer model: {model_name}")
model = SentenceTransformer(model_name)
dataset_path = "./dataset_WB.csv"
output_csv_path = 'outputs/new_test.csv'
max_corpus_size = 1024
print(f"Dataset path: {dataset_path}")
print(f"Max corpus size: {max_corpus_size}")
embedding_cache_path = 'outputs/new-{}-dataset-embeddings-{}-size-{}.pkl'.format(output_csv_path.split('.')[0].replace('/','_'), model_name.replace('/', '_'), max_corpus_size)
print(f"Embedding cache path: {embedding_cache_path}")

print(f"Results will be written to {output_csv_path}")
embedding_size = 768
top_k_hits = 129
n_clusters = 2
print(f"Embedding size: {embedding_size}")
print(f"Top k hits: {top_k_hits}")
print(f"Number of clusters: {n_clusters}")
print("Initializing FAISS index...")
quantizer = faiss.IndexFlatIP(embedding_size)
index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
index.nprobe = 3


# Function to read the dataset with two columns (instruction and tag)
def read_stories(file_path, max_size):
    print(f"Reading stories from {file_path}...")
    corpus_sentences = []
    # tags = []
    with open(file_path, 'r', encoding='utf8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            # Expecting: column 0 = tag, column 1 = instruction
            if len(row) < 2:
                # Skip rows that don’t have both tag and instruction
                continue
            instruction = row[1].strip()   # Second column holds the instruction
            corpus_sentences.append(instruction)
            # tags.append(row[0].strip())   # Tag can be stored later if needed
            if len(corpus_sentences) >= max_size:
                break
    print(f"Read {len(corpus_sentences)} unique entries")
    return list(zip(corpus_sentences))  # Return both instructions and tags


# If embeddings don't exist, create them
if not os.path.exists(embedding_cache_path):
    print("Embedding cache not found. Creating new embeddings...")
    corpus_data = read_stories(dataset_path, max_corpus_size)
    corpus_sentences = [data[0] for data in corpus_data]  # List of instructions
    # tags = [data[1] for data in corpus_data]              # List of tags
    print("Encoding the corpus. This might take a while...")
    corpus_embeddings = []
    for sentence in tqdm(corpus_sentences, desc="Encoding sentences"):
        corpus_embeddings.append(model.encode(sentence, show_progress_bar=True, convert_to_numpy=True))
    corpus_embeddings = np.array(corpus_embeddings)
    print("Storing embeddings on disk...")
    os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences,  'embeddings': corpus_embeddings}, fOut)
    print("Embeddings stored successfully")
else:
    print("Loading pre-computed embeddings from disk...")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        # tags = cache_data['tags']
        corpus_embeddings = cache_data['embeddings']
    print("Pre-computed embeddings loaded successfully")

# Creating the FAISS index
# print("Creating FAISS index...")
# corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
# print("Training the index...")
index.train(corpus_embeddings)
print("Adding embeddings to the index...")

for i in tqdm(range(0, len(corpus_embeddings), 1000), desc="Adding to FAISS index"):
    index.add(corpus_embeddings[i:i+1000])
print("FAISS index created and populated")
# Start processing questions and finding nearest neighbors
print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")


with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Question', 'Result', 'Score'])
    print("Starting the search process...")
    # Loop over all sentences in the corpus (as questions)
    for i in tqdm(range(len(corpus_sentences)), desc="Processing questions"):
        inp_question = corpus_sentences[i]
        # inp_tag = tags[i]  # Tag for the query
        print(f"\nProcessing question {i+1}/{len(corpus_sentences)}")
        print(f"Input question: {inp_question}")
        print("Encoding the question...")
        question_embedding = model.encode(inp_question)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        print("Searching in FAISS index...")
        start_time = time.time()
        distances, corpus_ids = index.search(question_embedding, top_k_hits)
        end_time = time.time()
        # Preparing the hits (results from FAISS)
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        print(f"Search completed in {end_time-start_time:.3f} seconds")
        print("Results:")
        for hit in hits[1:top_k_hits]:  # Skip the first hit because it will be the query itself
            result_sentence = corpus_sentences[hit['corpus_id']]
            # result_tag = tags[hit['corpus_id']]  # Get the tag of the result
            # Write to CSV with query tag and result tag
            csvwriter.writerow([inp_question, result_sentence, hit['score']])
            print(f"\t{hit['score']:.3f}\t{result_sentence}")
print("Search process completed. Results saved to: ", output_csv_path)
