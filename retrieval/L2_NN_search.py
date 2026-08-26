import os
import pickle
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import traceback
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook

try:
    print("Starting the script...")

    model_name = 'all-mpnet-base-v2'
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    excel_path = "./10_same_with_nn.xlsx"  
    max_corpus_size = 1025

    print(f"Excel workbook path: {excel_path}")
    print(f"Max corpus size: {max_corpus_size}")

    embedding_cache_path = 'outputs/new3-excel-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)
    print(f"Embedding cache path: {embedding_cache_path}")

    embedding_size = 768
    top_k_hits = 2
    n_clusters = 7

    print(f"Embedding size: {embedding_size}")
    print(f"Top k hits: {top_k_hits}")
    print(f"Number of clusters: {n_clusters}")

    print("Initializing FAISS index...")
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 3

    def read_instructions_from_excel(file_path, max_size):
        print(f"Reading instructions from Excel: {file_path}...")
        # Read the Excel file
        try:
            df = pd.read_excel(file_path)
            # Excel columns are 1-indexed in user terms but 0-indexed in pandas
            # So column 6 is index 5
            if df.shape[1] < 6:
                print(f"Warning: Excel file has only {df.shape[1]} columns, but we need at least 6")
                return []
                
            # Get values from column 6 (index 5)
            instructions = df.iloc[:, 7].dropna().tolist()
            instructions = [str(instr).strip() for instr in instructions if str(instr).strip()]
            
            # Limit corpus size
            if len(instructions) > max_size:
                instructions = instructions[:max_size]
                
            print(f"Read {len(instructions)} unique instructions")
            return instructions
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return []

    # If embeddings don't exist, create them
    if not os.path.exists(embedding_cache_path):
        print("Embedding cache not found. Creating new embeddings...")
        corpus_sentences = read_instructions_from_excel(excel_path, max_corpus_size)
        
        if not corpus_sentences:
            raise ValueError("No instructions found in the Excel file")
            
        print("Encoding the corpus. This might take a while...")
        corpus_embeddings = []
        for sentence in tqdm(corpus_sentences, desc="Encoding sentences"):
            corpus_embeddings.append(model.encode(sentence, show_progress_bar=False, convert_to_numpy=True))
        corpus_embeddings = np.array(corpus_embeddings)

        print("Storing embeddings on disk...")
        os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
        print("Embeddings stored successfully")
    else:
        print("Loading pre-computed embeddings from disk...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']
        print("Pre-computed embeddings loaded successfully")

    # Creating the FAISS index
    print("Creating FAISS index...")
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
    print("Training the index...")
    index.train(corpus_embeddings)
    print("Adding embeddings to the index...")
    for i in tqdm(range(0, len(corpus_embeddings), 1000), desc="Adding to FAISS index"):
        index.add(corpus_embeddings[i:i+1000])
    print("FAISS index created and populated")

    # Start processing questions and finding nearest neighbors
    print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")

    # Process all instructions and update the Excel file
    print("Starting the search process...")
    
    # Prepare results dictionary to store nearest neighbors
    results = {}
    
    # Loop over all sentences in the corpus
    for i in tqdm(range(len(corpus_sentences)), desc="Processing instructions"):
        inp_instruction = corpus_sentences[i]

        print(f"\nProcessing instruction {i+1}/{len(corpus_sentences)}")
        print(f"Input instruction: {inp_instruction}")

        print("Encoding the instruction...")
        instruction_embedding = model.encode(inp_instruction)
        instruction_embedding = instruction_embedding / np.linalg.norm(instruction_embedding)
        instruction_embedding = np.expand_dims(instruction_embedding, axis=0)

        print("Searching in FAISS index...")
        start_time = time.time()
        distances, corpus_ids = index.search(instruction_embedding, top_k_hits)
        end_time = time.time()

        # Preparing the hits (results from FAISS)
        hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)

        print(f"Search completed in {end_time-start_time:.3f} seconds")
        
        # Skip the first hit (it's the instruction itself)
        if len(hits) > 1:
            result_instruction = corpus_sentences[hits[1]['corpus_id']]
            result_score = hits[1]['score']
            results[inp_instruction] = (result_instruction, result_score)
            print(f"\tNearest neighbor: {result_instruction} (Score: {result_score:.3f})")
    
    # Update the Excel file with results
    print("Updating Excel file with nearest neighbors...")
    
    try:
        # Load the workbook while preserving existing content
        workbook = load_workbook(excel_path)
        sheet = workbook.active
        
        # Find the last row and column
        max_row = sheet.max_row
        
        # Update column 8 (index G in Excel) with nearest neighbors
        updated_count = 0
        
        for row in range(1, max_row + 1):
            # Get value from column 6 (index F in Excel)
            instruction_cell = sheet.cell(row=row, column=6)
            
            if instruction_cell.value and str(instruction_cell.value).strip() in results:
                # Get the nearest neighbor and score
                nn_instruction, nn_score = results[str(instruction_cell.value).strip()]
                
                # Write nearest neighbor to column 8 (index H in Excel)
                sheet.cell(row=row, column=12).value = nn_instruction
                
                # Optionally write score to column 9 (index I in Excel)
                sheet.cell(row=row, column=13).value = f"{nn_score:.4f}"
                
                updated_count += 1
        
        # Save the updated workbook
        output_excel_path = excel_path.replace('.xlsx', '_with_nn_new.xlsx')
        workbook.save(output_excel_path)
        
        print(f"Updated {updated_count} rows with nearest neighbors")
        print(f"Results saved to: {output_excel_path}")
        
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        print(traceback.format_exc())

    print("Script completed successfully")

except Exception as e:
    print(f"An error occurred: {str(e)}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)