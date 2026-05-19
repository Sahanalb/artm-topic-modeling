# Agentic RAG Topic Modeling (ARTM)
### Amazon Musical Instruments Reviews

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: macOS M4](https://img.shields.io/badge/platform-macOS%20M4-lightgrey.svg)](https://www.apple.com/mac/)

> **Masters Term Paper** — Agentic Retrieval-Augmented Topic Modeling for E-Commerce Product Reviews

---

## Overview

**ARTM** is a novel topic modeling framework that combines:
- **FAISS** dense vector retrieval for evidence grounding
- **BERTopic** for initial semantic clustering
- **Flan-T5 LLM agent** for iterative propose → critique → refine labeling
- **MPS acceleration** for Apple Silicon (M1/M2/M3/M4)

Applied to **10,206 Amazon Musical Instruments reviews**, ARTM discovers interpretable product taxonomy topics through an agentic feedback loop — the first system to combine RAG-based evidence retrieval with LLM self-critique for unsupervised topic modeling.

---

## Results

| Model | Topics | CV Coherence | NPMI | Diversity | Sem. Sep. |
|---|---|---|---|---|---|
| BERTopic (baseline) | 9 | 0.4786 | -0.0864 | 0.9041 | 0.8249 |
| **ARTM (ours)** | **8** | **0.4839** | **-0.0577** | **0.9104** | 0.8069 |

### Discovered Topics

| # | Topic | Top Keywords | Score |
|---|---|---|---|
| 1 | Guitar tuners | snark, tuner, clip, chromatic, accurate | 0.531 |
| 2 | Cables | cable, cord, adapter, patch, noise | 0.261 |
| 3 | Delay pedal | delay, pedal, reverb, distortion, amp | 0.367 |
| 4 | Music stand | stand, folding, portable, stable, sheet | 0.475 |
| 5 | Cleaning guitars | cloth, polish, strings, care, wipe | 0.428 |
| 6 | Pop filters | filter, pop, mic, recording, studio | 0.188 |
| 7 | Bench | bench, height, piano, adjustable, padded | 0.249 |
| 8 | Dunlop picks | dunlop, picks, finger, ring, slide | 0.374 |

---

## Repository Structure

```
artm-topic-modeling/
│
├── README.md                          ← this file
├── LICENSE                            ← MIT license
├── requirements.txt                   ← all dependencies
│
├── data/
│   └── README_data.md                 ← how to download the dataset
│
├── src/
│   ├── agentic_rag_amazon_v3.py       ← main ARTM pipeline
│   ├── baseline_bertopic.py           ← BERTopic baseline
│   ├── evaluation_framework.py        ← all evaluation metrics
│   
│
├── outputs/
│   ├── final_topics.txt               ← discovered topic labels
│   ├── evaluation_results.json        ← quantitative metrics
│   ├── baseline_results.json          ← baseline metrics
│   
│
└── paper/
    └── artm_paper.docx                ← full research paper
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sahanalb/artm-topic-modeling.git
cd artm-topic-modeling
```

### 2. Set up environment

```bash
python3 -m venv artm_env
source artm_env/bin/activate        # Mac/Linux
pip install -r requirements.txt
```

### 3. Download the dataset

The Amazon Musical Instruments dataset is available from the
[UCSD Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) page.

Download `Musical_Instruments_5.json` and run the conversion script:

```bash
mkdir data
cp Musical_Instruments_5.json data/
python src/convert_to_jsonl.py
```

This produces `data/amazon_musical_instruments_reviews.jsonl` (10,206 reviews).

### 4. Run the ARTM pipeline

```bash
python src/agentic_rag_amazon_v3.py
```

Expected output (Apple M4, ~25 minutes):
```
ARTM Pipeline v3 — Amazon Musical Instruments
Loaded 10,206 reviews.
Encoding reviews... (2-3 minutes)
FAISS index: 10,206 vectors, dim=384
BERTopic found 9 valid topics.
── Agentic iteration 1/3 ──
  Topic  0: 'Guitar tuners'  | score=0.810
  ...
cv_coherence  : 0.4839
topic_diversity: 0.9104
```

### 5. Run the baseline

```bash
python src/baseline_bertopic.py
```

### 6. Run evaluation

```bash
python src/evaluation_framework.py
```

---

## System Requirements

| Component | Requirement |
|---|---|
| Python | 3.11+ |
| RAM | 8GB minimum, 16GB recommended |
| GPU | Apple MPS (M1/M2/M3/M4) or CPU |
| Storage | ~3GB (models + data) |
| Time | ~25 min (M4 MPS) / ~60 min (CPU) |

---

## Dependencies

Key packages (see `requirements.txt` for full list):

```
sentence-transformers>=2.7.0
faiss-cpu>=1.8.0
bertopic>=0.16.0
transformers>=4.40.0
torch>=2.3.0
umap-learn>=0.5.6
hdbscan>=0.8.38
gensim>=4.3.0
scikit-learn>=1.4.0
```

---

## Algorithm

```
Algorithm 1: ARTM
Input:  D (reviews), ε (embedder), A (LLM agent), T (iterations), τ (threshold)
Output: Θ (topic set with labels)

1.  E  ← ε(D)                    // embed all reviews
2.  F  ← FAISS_index(E)          // build inner-product index
3.  Θ₀ ← BERTopic(D, E)          // seed topic clusters
4.  t  ← 0
5.  while t < T:
6.    for each topic θᵢ in Θₜ:
7.      Rᵢ ← FAISS_retrieve(F, θᵢ, k=30)    // retrieve evidence
8.      lᵢ ← A.propose(θᵢ, Rᵢ)              // LLM labels topic
9.      sᵢ ← A.critique(lᵢ, Rᵢ, Θₜ)         // embedding-based score
10.     if sᵢ < τ: lᵢ ← A.refine(lᵢ, Rᵢ)   // refine weak labels
11.   Θₜ₊₁ ← deduplicate(Θₜ, cos=0.65)
12.   t ← t + 1
13. return Θₜ
```

---

## Citation

If you use this code or paper in your research, please cite:

```bibtex
@article{Sahanalb2026artm,
  title   = {Agentic Retrieval-Augmented Topic Modeling for
             Amazon Musical Instruments Reviews},
  author  = {Sahanalb},
  year    = {2026},
  url     = {https://github.com/Sahanalb/artm-topic-modeling}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Amazon Review Dataset: He & McAuley (2016)
- BERTopic: Grootendorst (2022)
- Sentence Transformers: Reimers & Gurevych (2019)
- FAISS: Johnson et al. (2021)
- Reflexion (agentic inspiration): Shinn et al. (2023)
