"""
SSR pipeline for multimodal retrieval with CLIP.

Logic overview:
1) Load a CLIP model and choose CUDA/CPU automatically.
2) Read test/train JSONL files for each dataset.
3) Build one fused embedding per item (text + 4 * image) / 5.
4) Compute cosine similarities between every test item and every train item.
5) For each test item, keep top-k positive matches and save to JSONL.

Output format (one line per test sample):
{
  "index": <test_index>,
  "samples": [<train_index_1>, ...],
  "scores": [<similarity_1>, ...]
}
"""

import os
import torch
import clip
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import copy

# Import data utility functions
from utils.data_utils import get_item_data, DATASET_CONFIGS

# Determine the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the CLIP model and preprocessor
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model.eval()

def process_clip_embeddings(dataset_name: str, k: int = 10):
    """
        Processes CLIP embeddings for a given dataset, calculates similarity scores,
    and saves the top-k similar samples.
    Process one dataset end-to-end and store top-k train neighbors for each test item.

    Steps:
    - load dataset files
    - encode text/image with CLIP
    - fuse image-text features
    - compute cosine similarity matrix
    - export top-k indices and scores as JSONL
    """
    print(f"\n--- Processing dataset: {dataset_name} ---")

    base_data_path = f"data/{dataset_name}"
    image_base_path = f"{base_data_path}/images"
    test_jsonl_path = f"{base_data_path}/test.jsonl"
    train_jsonl_path = f"{base_data_path}/train.jsonl"
    result_path = f"SSR/{dataset_name}_SSR.jsonl"

    try:
        test_data = [json.loads(line) for line in open(test_jsonl_path, 'r').readlines()]
        train_data = [json.loads(line) for line in open(train_jsonl_path, 'r').readlines()]
    except FileNotFoundError as e:
        print(f"Error: Data files not found for {dataset_name}. Missing file: {e.filename}")
        return

    print(f"Loaded {len(test_data)} test items and {len(train_data)} train items for {dataset_name}.")

    # Embeddings for test items (query side).
    embeddings = []
    print(f"Generating embeddings for {dataset_name} test data...")
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Test Embeddings"):
        # Use get_item_data for unified access
        image_file_name, text_content, _ = get_item_data(item, dataset_name)
        if not image_file_name or not text_content:
            print(f"Warning: Missing image file name or text content for test item {idx}. Skipping.")
            continue

        image_file_path = os.path.join(image_base_path, image_file_name)

        try:
            image = Image.open(image_file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_file_path}. Skipping item.")
            continue
        except Exception as e:
            print(f"Warning: Could not open image {image_file_path}: {e}. Skipping item.")
            continue

        processed_image = preprocess(image).unsqueeze(0).to(device)
        tokenized_text = clip.tokenize([text_content], truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(processed_image).squeeze().detach().cpu().numpy()
            text_features = model.encode_text(tokenized_text).squeeze().detach().cpu().numpy()

        # Weighted fusion: image contributes 4x more than text.
        embedding = (text_features + image_features * 4) / 5
        embeddings.append(embedding)

    # Embeddings for train items (reference/candidate side).
    ref_embeddings = []
    print(f"Generating embeddings for {dataset_name} train data...")
    for idx, item in tqdm(enumerate(train_data), total=len(train_data), desc="Train Embeddings"):
        # Use get_item_data for unified access
        image_file_name, text_content, _ = get_item_data(item, dataset_name)
        if not image_file_name or not text_content:
            print(f"Warning: Missing image file name or text content for train item {idx}. Skipping.")
            continue

        image_file_path = os.path.join(image_base_path, image_file_name)

        try:
            image = Image.open(image_file_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found: {image_file_path}. Skipping item.")
            continue
        except Exception as e:
            print(f"Warning: Could not open image {image_file_path}: {e}. Skipping item.")
            continue

        processed_image = preprocess(image).unsqueeze(0).to(device)
        tokenized_text = clip.tokenize([text_content], truncate=True).to(device)

        with torch.no_grad():
            image_features = model.encode_image(processed_image).squeeze().detach().cpu().numpy()
            text_features = model.encode_text(tokenized_text).squeeze().detach().cpu().numpy()

        # Keep the same fusion rule for test/train consistency.
        embedding = (text_features + image_features * 4) / 5
        ref_embeddings.append(embedding)

    if not embeddings or not ref_embeddings:
        print(f"No embeddings generated for {dataset_name}. Exiting.")
        return

    embeddings_np = np.array(embeddings)
    ref_embeddings_np = np.array(ref_embeddings)

    print(f"Calculating similarity scores for {dataset_name}...")
    similarity_scores = np.zeros((len(embeddings_np), len(ref_embeddings_np)))

    # Cosine similarity = dot(a, b) / (||a|| * ||b||), vectorized for speed.
    dot_products = np.dot(embeddings_np, ref_embeddings_np.T)
    norms_embeddings = norm(embeddings_np, axis=1, keepdims=True)
    norms_ref_embeddings = norm(ref_embeddings_np, axis=1, keepdims=True).T

    norms_embeddings[norms_embeddings == 0] = 1
    norms_ref_embeddings[norms_ref_embeddings == 0] = 1
    similarity_scores = dot_products / (norms_embeddings * norms_ref_embeddings)
    # Numerical safety: keep values in valid cosine range.
    similarity_scores = np.clip(similarity_scores, -1.0, 1.0)
    # Exclude exact self-like/perfect matches from candidate selection.
    similarity_scores[similarity_scores >= 1.0] = 0.0
    
    # Deep copy the similarity_scores array to prevent modification issues during top-k extraction
    similarity_scores_copy = copy.deepcopy(similarity_scores)


    results = []
    print(f"Extracting top-{k} similar samples for {dataset_name}...")
    for i in tqdm(range(len(embeddings_np)), desc="Top-K Extraction"):
        samples = []
        scores = []
        current_scores = similarity_scores_copy[i]

        # Greedy top-k extraction without replacement for the current test item.
        for _ in range(k):
            if np.max(current_scores) <= 0:
                break
            j = int(np.argmax(current_scores))
            samples.append(j)
            # Convert numpy float to standard Python float for JSON serialization
            scores.append(float(current_scores[j]))
            current_scores[j] = -1

        results.append({
            "index": i, # Index of the test item
            "samples": samples, # Indices of the top-k train items
            "scores": scores, # Corresponding similarity scores
        })

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        for result_item in results:
            json.dump(result_item, f)
            f.write("\n")
    print(f"Results saved to {result_path}")
    print(f"--- Finished processing {dataset_name} ---")

if __name__ == "__main__":
    datasets_to_process = ["FHM", "HarM", "MAMI"]
    for dataset in datasets_to_process:
        process_clip_embeddings(dataset)
