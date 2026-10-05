# Temporal Dense Retrieval (TDR): Time-Aware Semantic Search

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-orange.svg)](https://sbert.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
Standard dense retrieval models often fail in dynamic, time-sensitive environments (e.g., news aggregation, financial intelligence) because they treat the embedding space as static, treating outdated and fresh information equally[cite: 9]. This project bridges the gap between traditional Temporal Information Retrieval (TIR) and modern Neural Ranking by introducing the **Temporal Dense Retrieval (TDR) framework**[cite: 9]. TDR integrates a parameterized exponential time-decay function directly into the semantic scoring pipeline of Sentence-BERT (SBERT)[cite: 9]. This allows the system to prioritize highly recent content while preserving deep semantic understanding—critical for both production-grade search engines and advanced academic research in Information Retrieval[cite: 9].

## 🚀 Business & Academic Value
* **For Industry:** Provides a lightweight, highly scalable post-processing re-ranking layer that eliminates the need to continuously fine-tune or pre-train large language models to maintain temporal awareness.
* **For Research:** Establishes a modular baseline for time-aware semantic search, proving that dense embeddings can be successfully modulated using temporal heuristics without degrading topical relevance[cite: 9].

## 🏗️ Architecture & Methodology
The TDR framework is designed as a decoupled pipeline to ensure computational efficiency:

1. **Semantic Encoding:** Both the user query ($q$) and document corpus ($D$) are encoded into a 384-dimensional dense vector space using a pre-trained SBERT model (`all-MiniLM-L6-v2`)[cite: 9].
2. **Dense Retrieval (Top-K):** The system retrieves top candidates using exact or approximate nearest neighbor search based on Cosine Similarity:
   $$S_{sem}(q,d) = \frac{v_q \cdot v_d}{\vert{}\vert{}v_q\vert{}\vert{} \vert{}\vert{}v_d\vert{}\vert{}}$$
3. **Temporal Decay Modeling:** An exponential decay filter models the diminishing utility of information over time[cite: 9]. Where $\Delta t$ is the document age in days and $\lambda$ is the decay rate tuning parameter[cite: 9]:
   $$S_{temp}(d) = e^{-\lambda \cdot \Delta t}$$
4. **Hybrid Score Fusion:** A final re-ranking score linearly interpolates semantic relevance and temporal freshness[cite: 9]. The weight parameter $\alpha$ allows the model to shift dynamically between "breaking news" and "archival research" intents[cite: 9]:
   $$S_{final}(q,d) = \alpha \cdot S_{sem}(q,d) + (1-\alpha) \cdot S_{temp}(d)$$

## 🛠️ Tech Stack & Infrastructure
* **Core Language:** Python 3.10[cite: 9]
* **Modeling & Embeddings:** `sentence-transformers`, Sentence-BERT (SBERT)[cite: 9]
* **Compute Infrastructure:** Optimized for NVIDIA Tesla T4 GPU (Google Colab)[cite: 9]
* **Data Processing:** Pandas, NumPy
* **Dataset:** CNN/DailyMail benchmark (500-document evaluation subset with simulated temporal distributions spanning 2020–2025)[cite: 9]

## 📊 Evaluation & Key Results
The framework was empirically validated against a simulated multi-year news archive[cite: 9]. 
* **Recency Promotion:** TDR successfully pushed structurally relevant but outdated documents down the ranking while surfacing fresh documents matching the query intent[cite: 9].
* **Quantitative Shift:** In a top-5 retrieval test for AI advancements, a highly recent document was dynamically boosted to rank #1—elevating its base semantic score of $0.171$ to a dominant TDR score of $0.403$[cite: 9].
