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



### CNN & DAILY MAIL



!pip install sentence-transformers pandas datasets

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from datetime import datetime, timedelta
import random

# 1. LOAD REAL DATA (CNN/DailyMail)
print("Loading CNN/DailyMail dataset (this may take a minute)...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:500]") # Taking 500 articles for speed

# 2. PREPROCESS (Simulate Dates)
# The dataset doesn't have metadata dates easily accessible, so we will simulate
# a realistic distribution of dates for this experiment (2020-2026).
docs = []
base_date = datetime(2026, 1, 15)
print("Preprocessing data...")

for i, item in enumerate(dataset):
    # Simulate a date: mostly recent, some old
    days_back = random.randint(0, 2000) # Up to 5-6 years back
    doc_date = base_date - timedelta(days=days_back)

    docs.append({
        "id": item['id'],
        "text": item['article'][:500], # First 500 chars for embedding
        "title": item['article'][:100].split('\n')[0], # Pseudo-title
        "date": doc_date
    })

df = pd.DataFrame(docs)
current_date = base_date

# 3. MODEL SETUP
model = SentenceTransformer('all-MiniLM-L6-v2')

# 4. SCORING FUNCTIONS
def get_scores(query, doc_texts, doc_dates):
    # Semantic
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(doc_texts, convert_to_tensor=True)
    from sentence_transformers.util import cos_sim
    sem_scores = cos_sim(query_emb, doc_embs)[0].cpu().numpy()

    # Temporal
    deltas = np.array([(current_date - d).days for d in doc_dates])
    deltas = np.maximum(deltas, 0)
    temp_scores = np.exp(-0.002 * deltas) # Lambda = 0.002

    return sem_scores, temp_scores

# 5. RUN EXPERIMENT ON A QUERY
query = "recent advancements in artificial intelligence"
sem_scores, temp_scores = get_scores(query, df['text'].tolist(), df['date'].tolist())

alpha = 0.7
df['semantic_score'] = sem_scores
df['temporal_score'] = temp_scores
df['final_score'] = (alpha * sem_scores) + ((1-alpha) * temp_scores)

# 6. GENERATE LATEX TABLE
top_k = df.sort_values(by='final_score', ascending=False).head(5)

print("\n--- COPY THIS INTO PAPER BY LATEX (TABLE) ---")
print("\\begin{table}[h!]")
print("\\centering")
print("\\caption{Top-5 Retrieved Documents for query: '" + query + "' using TDR}")
print("\\begin{tabular}{|p{6cm}|c|c|c|}")
print("\\hline")
print("\\textbf{Document Content (Snippet)} & \\textbf{Date} & \\textbf{Sem. Score} & \\textbf{TDR Score} \\\\")
print("\\hline")
for _, row in top_k.iterrows():
    title_snip = row['title'][:60] + "..."
    date_str = row['date'].strftime("%Y-%m-%d")
    print(f"{title_snip} & {date_str} & {row['semantic_score']:.3f} & \\textbf{{{row['final_score']:.3f}}} \\\\")
    print("\\hline")
print("\\end{tabular}")
print("\\end{table}")
