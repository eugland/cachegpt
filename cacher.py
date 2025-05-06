from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import numpy as np
import faiss

# Model names
distilled_model_name = "distilgpt2"
parent_model_name = "gpt2-medium"

# Load models and tokenizers
distilled_tokenizer = AutoTokenizer.from_pretrained(distilled_model_name)
distilled_model = AutoModelForCausalLM.from_pretrained(distilled_model_name)

parent_tokenizer = AutoTokenizer.from_pretrained(parent_model_name)
parent_model = AutoModelForCausalLM.from_pretrained(parent_model_name)

# Fix tokenizer pad token issues
distilled_tokenizer.pad_token = distilled_tokenizer.eos_token
parent_tokenizer.pad_token = parent_tokenizer.eos_token

# Load embedding model for semantic cache
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index and cache mapping
dimension = 384
index = faiss.IndexFlatL2(dimension)
query_to_response = []

CONFIDENCE_THRESHOLD = 0.7

def estimate_confidence(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt", return_attention_mask=True, padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits[0, -1], dim=-1)
        return torch.max(probs).item()

def semantic_search(query, k=1):
    query_vec = embedder.encode([query])
    if index.ntotal == 0:
        return None
    distances, indices = index.search(np.array(query_vec), k)
    if distances[0][0] < 0.5:
        return query_to_response[indices[0][0]]
    return None

def cache_response(query, response):
    query_vec = embedder.encode([query])
    index.add(np.array(query_vec))
    query_to_response.append(response)

def generate_response(model, tokenizer, query):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(
        inputs["input_ids"],
        max_length=100,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def handle_query(query):
    cached = semantic_search(query)
    if cached:
        return f"[CACHE HIT] {cached}"

    confidence = estimate_confidence(distilled_model, distilled_tokenizer, query)

    print(f"Confidence: {confidence:.2f}")
    if confidence > CONFIDENCE_THRESHOLD:
        response = generate_response(distilled_model, distilled_tokenizer, query)
        cache_response(query, response)
        return f"[DISTILLED] {response}"
    else:
        response = generate_response(parent_model, parent_tokenizer, query)
        cache_response(query, response)
        return f"[PARENT] {response}"

# # Run an example
# example_query = "What causes lightning?"
# print(f"Query: {example_query}")
# result = handle_query(example_query)
# print(result)

# 🔹 Option 1: Embedding-Based Semantic Similarity Heuristic (No LLM)
# Use only the embedding model (all-MiniLM-L6-v2) to compute similarity to previously seen queries. If the new query is semantically close to a cached query, infer high confidence.

# ✅ Pros:
# Very fast (no LLM forward pass)

# Works offline once embedded

# Great for repeated or FAQ-style queries

# 🔧 How:
# python
# Copy
# Edit
# def estimate_confidence_from_cache(query, threshold=0.5):
#     if index.ntotal == 0:
#         return 0.0
#     query_vec = embedder.encode([query])
#     distances, _ = index.search(np.array(query_vec), 1)
#     similarity_score = 1.0 - distances[0][0]  # Lower distance = higher similarity
#     return similarity_score if similarity_score > threshold else 0.0
# You can then use this to short-circuit before running the LLM:

# python
# Copy
# Edit
# confidence = estimate_confidence_from_cache(query)
# if confidence > threshold:
#     return f"[CACHE-SIMILARITY] {cached_response}"
# 🔹 Option 2: Use a Fast Classifier or Routing Model
# Train a mini classifier (e.g., logistic regression) on query embeddings labeled with “distilled succeeded” or not. Then use it to quickly predict if the edge model is likely to succeed.

# ✅ Pros:
# Customizable, tunable

# Works well with labeled data

# ❌ Requires:
# A dataset of queries and confidence results

# Some training effort

# 🔹 Option 3: Use a Precomputed Token Entropy Table
# For very domain-specific tasks (e.g., customer support, finance), build a token-level entropy estimate for common query patterns.

# But this is complex and brittle outside of controlled datasets.

