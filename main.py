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
