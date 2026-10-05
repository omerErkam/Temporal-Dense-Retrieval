# Temporal Dense Retrieval (TDR)

## Project Overview
Standard dense retrieval models often fail in dynamic environments like news aggregation because they ignore the temporal dimension of a document's relevance, treating outdated and fresh information equally[cite: 9]. This project introduces a Temporal Dense Retrieval (TDR) framework that resolves this by integrating an exponential time-decay function into the semantic similarity score, ensuring fresh content is prioritized without sacrificing underlying semantic understanding[cite: 9].

## Tech Stack
* **Language:** Python 3.10[cite: 9]
* **Libraries & Models:** `sentence-transformers`, Sentence-BERT (SBERT), `all-MiniLM-L6-v2`[cite: 9]
* **Infrastructure:** Google Colab, NVIDIA Tesla T4 GPU[cite: 9]
* **Dataset:** CNN/DailyMail (500-document subset utilizing a simulated temporal distribution from 2020 to 2025)[cite: 9]

## Methodology
1. **Semantic Encoding:** A pre-trained SBERT model maps both the user queries and corpus documents into a shared 384-dimensional dense vector space[cite: 9].
2. **Dense Retrieval:** The baseline topical relevance is calculated by measuring the cosine similarity between the query and document embeddings[cite: 9].
3. **Temporal Re-ranking:** A temporal decay filter calculates the age of each document in days and applies an exponential decay function to mathematically model the diminishing relevance of older news events[cite: 9].
4. **Hybrid Score Fusion:** A weighted linear interpolation strategy fuses the semantic and temporal scores into a final ranking metric, allowing the system to adapt to different query intents by adjusting tuning parameters[cite: 9].

## Results & Evaluation
* The TDR framework successfully prioritized recent, relevant documents over outdated ones in simulated temporal environments[cite: 9].
* In a top-5 retrieval test for a specific query, the model effectively recognized the recency of a document, boosting it to the top rank by elevating a base semantic score of 0.171 to a final TDR score of 0.403[cite: 9].
