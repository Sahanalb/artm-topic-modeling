"""
Baseline Evaluation — BERTopic Only (No Agentic Loop)
======================================================
Run this SEPARATELY from the main ARTM pipeline.
Results go into Table 1 of your paper as the baseline row.

Usage:
    python baseline_bertopic.py
"""

import json, re
from pathlib import Path
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
nltk.download("stopwords", quiet=True)

# ── Config — must match your ARTM v3 settings exactly ─────────────────────────
DATA_FILE   = "data/amazon_real_musical_instruments_reviews.jsonl"
NUM_TOPICS  = 10
RANDOM_SEED = 42
BATCH_SIZE  = 64

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_reviews(path):
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get("review", "").strip()
                if len(text) > 40:
                    reviews.append(text)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(reviews):,} reviews.", flush=True)
    return reviews

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def build_embeddings(docs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding reviews...", flush=True)
    embs = model.encode(
        docs, batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    return model, embs

def run_bertopic_baseline(docs, embs, embedder):
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )
    umap_model = UMAP(
        n_neighbors=15, n_components=5,
        min_dist=0.0, metric="cosine",
        random_state=RANDOM_SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=MaximalMarginalRelevance(diversity=0.4),
        nr_topics=NUM_TOPICS,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs, embs)
    info = topic_model.get_topic_info()

    topic_words = {}
    topic_labels = {}
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        words_scores = topic_model.get_topic(tid)
        words = [w for w, _ in words_scores]
        # Label = top 3 keywords joined
        label = " / ".join(words[:3])
        topic_words[tid] = words
        topic_labels[tid] = label

    print(f"BERTopic found {len(topic_words)} topics.", flush=True)
    return topic_words, topic_labels

def evaluate(topic_words_dict, topic_labels_dict, docs, embedder):
    tokenized = [d.lower().split() for d in docs]
    dictionary = Dictionary(tokenized)
    valid_words = set(dictionary.token2id.keys())

    # Filter keywords to corpus vocabulary
    topic_words_list = []
    for words in topic_words_dict.values():
        filtered = [w for w in words if w in valid_words]
        if len(filtered) >= 2:
            topic_words_list.append(filtered)

    if len(topic_words_list) < 2:
        return {"error": "too few valid topics"}

    cv = CoherenceModel(
        topics=topic_words_list, texts=tokenized,
        dictionary=dictionary, coherence="c_v"
    ).get_coherence()

    npmi = CoherenceModel(
        topics=topic_words_list, texts=tokenized,
        dictionary=dictionary, coherence="c_npmi"
    ).get_coherence()

    all_words = [w for t in topic_words_list for w in t[:25]]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0.0

    # Semantic separation
    labels = list(topic_labels_dict.values())
    label_embs = embedder.encode(labels, normalize_embeddings=True)
    sim_mat = cosine_similarity(label_embs)
    n = len(labels)
    if n > 1:
        off_diag = (sim_mat.sum() - np.trace(sim_mat)) / (n * (n - 1))
        semantic_sep = round(1.0 - float(off_diag), 4)
    else:
        semantic_sep = 1.0

    # Semantic validity
    rev_embs = embedder.encode(docs[:200], normalize_embeddings=True)
    valid = sum(
        1 for t_emb in label_embs
        if cosine_similarity([t_emb], rev_embs)[0].max() >= 0.30
    )
    semantic_validity = round(valid / len(label_embs), 4)

    return {
        "n_topics": len(topic_words_list),
        "cv_coherence": round(float(cv), 4),
        "npmi_coherence": round(float(npmi), 4),
        "topic_diversity": round(diversity, 4),
        "semantic_separation": semantic_sep,
        "semantic_validity": semantic_validity,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BASELINE — BERTopic Only (no agentic loop)")
    print("=" * 60, flush=True)

    raw_docs = load_reviews(DATA_FILE)
    clean_docs = [clean_text(r) for r in raw_docs]

    embedder, embs = build_embeddings(clean_docs)
    topic_words, topic_labels = run_bertopic_baseline(
        clean_docs, embs, embedder
    )

    metrics = evaluate(topic_words, topic_labels, clean_docs, embedder)

    # Print topics
    print("\n" + "=" * 60)
    print("BASELINE TOPICS (BERTopic keywords — no LLM labeling)")
    print("=" * 60)
    for i, (tid, label) in enumerate(topic_labels.items(), 1):
        print(f"  {i:>2}. {label}")

    # Print metrics
    print("\n" + "=" * 60)
    print("BASELINE METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")

    # Save
    Path("outputs_v3").mkdir(exist_ok=True)
    with open("outputs_v3/baseline_results.json", "w") as f:
        json.dump({
            "model": "BERTopic (baseline)",
            "metrics": metrics,
            "topics": [
                {"topic_id": tid, "label": label,
                 "keywords": topic_words[tid][:10]}
                for tid, label in topic_labels.items()
            ]
        }, f, indent=2)

    print("\nBaseline results saved to outputs_v3/baseline_results.json")

    # Print comparison table
    print("\n" + "=" * 60)
    print("TABLE 1 PREVIEW — copy these numbers into your paper")
    print("=" * 60)
    print(f"{'Model':<25} {'Topics':>6} {'CV':>7} {'NPMI':>8} "
          f"{'Diversity':>10} {'Sem.Sep':>8}")
    print("-" * 65)
    print(f"{'BERTopic (baseline)':<25} "
          f"{metrics['n_topics']:>6} "
          f"{metrics['cv_coherence']:>7.4f} "
          f"{metrics['npmi_coherence']:>8.4f} "
          f"{metrics['topic_diversity']:>10.4f} "
          f"{metrics['semantic_separation']:>8.4f}")
    print(f"{'ARTM (ours)':<25} "
          f"{'8':>6} "
          f"{'0.4839':>7} "
          f"{'-0.0577':>8} "
          f"{'0.9104':>10} "
          f"{'0.8069':>8}")
    print("=" * 60)
    print("\nDone!", flush=True)

if __name__ == "__main__":
    main()
