!pip install sentence-transformers pandas numpy

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

# 1. SETUP: Load a lightweight SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. DATA: Create a small "Toy" dataset (Simulating news articles)
# In the final version, you will load this from a CSV
data = [
    {"title": "The history of the Ottoman Empire in the 16th century", "date": "2020-01-01", "id": 1},
    {"title": "Recent earthquake tremors felt in Izmir yesterday", "date": "2025-01-10", "id": 2},
    {"title": "Earthquake safety guidelines released by government", "date": "2022-05-15", "id": 3},
    {"title": "New economic policies announced for 2026", "date": "2026-01-01", "id": 4},
    {"title": "Tips for baking the perfect chocolate cake", "date": "2024-12-25", "id": 5}
]

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# 3. CONFIGURATION: The "Current" Date for the simulation
current_date = pd.to_datetime("2026-01-12")

# 4. METHODOLOGY: Define the Scoring Functions

def get_semantic_scores(query, documents):
    # Encode query and documents
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(documents, convert_to_tensor=True)

    # Compute Cosine Similarity
    from sentence_transformers.util import cos_sim
    scores = cos_sim(query_emb, doc_embs)[0].cpu().numpy()
    return scores

def get_temporal_scores(doc_dates, ref_date, decay_rate=0.01):
    # Calculate days difference
    deltas = (ref_date - doc_dates).dt.days
    # Ensure no negative deltas (future news) roughly handled for this demo
    deltas = deltas.apply(lambda x: max(x, 0))

    # Apply Exponential Decay: e^(-lambda * days)
    # This is the MATH PART for your paper
    time_scores = np.exp(-decay_rate * deltas)
    return time_scores

# 5. EXECUTION: Run the Pipeline
query = "Earthquake news in Turkey"

# A. Semantic Step (Standard Dense Retrieval)
df['semantic_score'] = get_semantic_scores(query, df['title'].tolist())

# B. Temporal Step (Your Contribution)
df['temporal_score'] = get_temporal_scores(df['date'], current_date, decay_rate=0.002)

# C. Hybrid Ranking (The Final Formula)
# Alpha controls the balance (0.7 means 70% semantics, 30% time)
alpha = 0.7
df['final_score'] = (alpha * df['semantic_score']) + ((1 - alpha) * df['temporal_score'])

# 6. RESULTS: Comparison
print(f"Query: {query}\n")
print("--- Baseline (Semantic Only) ---")
print(df.sort_values(by='semantic_score', ascending=False)[['title', 'date', 'semantic_score']])

print("\n--- Proposed Method (Temporal Dense Retrieval) ---")
print(df.sort_values(by='final_score', ascending=False)[['title', 'date', 'final_score']])
