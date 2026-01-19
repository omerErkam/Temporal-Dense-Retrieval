# Temporal Dense Retrieval (TDR)

**Author:** Ömer Erkam İbiş  
**Course:** CENG 543 - Information Retrieval (Fall 2025-2026)  
**Institute:** İzmir Institute of Technology  

## Overview
This project introduces a **Temporal Dense Retrieval** framework that adds a time-decay layer to standard semantic search models. It is designed to prioritize breaking news and recent events in Information Retrieval tasks.

## Methodology
The system uses a two-stage pipeline:
1. **Semantic Retrieval:** Uses `sentence-transformers` (SBERT) to find semantically similar documents.
2. **Temporal Re-ranking:** Applies an exponential decay function to the similarity scores based on the document's age.

## Installation
```bash
pip install sentence-transformers pandas numpy

## Usage
This script downloads a subset of the CNN/DailyMail dataset, simulates temporal distribution, and re-ranks results using the TDR framework.

```bash
python main.py
