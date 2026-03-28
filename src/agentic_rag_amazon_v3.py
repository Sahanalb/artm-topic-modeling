

import re, json, sys
from pathlib import Path

import numpy as np
import faiss
import torch
from tqdm import tqdm

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity

# ── Force stdout to flush immediately (fixes PyCharm output issues) ───────────
sys.stdout.reconfigure(line_buffering=True)

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_FILE      = "data/amazon_real_musical_instruments_reviews.jsonl"
OUTPUT_DIR     = Path("outputs_v3")
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "google/flan-t5-base"
NUM_TOPICS     = 10
TOP_K_RETRIEVE = 30
MAX_ITER       = 3
OVERLAP_SIM    = 0.75      # stricter deduplication (was 0.82)
BATCH_SIZE     = 64
RANDOM_SEED    = 42

# Define STOP_WORDS before any function uses it
STOP_WORDS = set(nltk_stopwords.words("english"))
STOP_WORDS -= {"no", "not", "very", "too", "but", "more", "most", "well"}


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_reviews(path: str) -> list:
    """Load ALL reviews — no artificial cap."""
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


# ── 2. Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light cleaning only — preserve meaningful tokens.
    Do NOT strip stopwords here; BERTopic's CountVectorizer handles that.
    Heavy stopword removal was the cause of 'believe ve', 'stays tune' in v1.
    """
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ── 3. Embeddings + FAISS ─────────────────────────────────────────────────────

def build_embeddings(docs: list, model_name: str = EMBED_MODEL) -> tuple:
    model = SentenceTransformer(model_name)
    print("Encoding reviews... (this takes 2-3 minutes)", flush=True)
    embs = model.encode(
        docs,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    return model, embs


def build_faiss_index(embs: np.ndarray) -> faiss.IndexFlatIP:
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"FAISS index: {index.ntotal:,} vectors, dim={dim}", flush=True)
    return index


def retrieve(query_text: str, embedder, index, docs: list,
             k: int = TOP_K_RETRIEVE) -> list:
    q = embedder.encode(
        [query_text], normalize_embeddings=True
    ).astype("float32")
    _, ids = index.search(q, k)
    return [docs[i] for i in ids[0] if 0 <= i < len(docs)]


# ── 4. BERTopic seed clustering ───────────────────────────────────────────────

def run_bertopic(docs: list, embs: np.ndarray,
                 embedder, n_topics: int = NUM_TOPICS) -> tuple:
    print("Running BERTopic clustering...", flush=True)

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,           # word must appear in at least 5 reviews
        max_df=0.90,        # ignore words in >90% of reviews (too common)
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
    representation_model = MaximalMarginalRelevance(diversity=0.4)

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        nr_topics=n_topics,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(docs, embs)
    info = topic_model.get_topic_info()

    topic_words = {}
    for tid in info["Topic"].tolist():
        if tid == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(tid)]
        if words:
            topic_words[tid] = words

    print(f"BERTopic found {len(topic_words)} valid topics.", flush=True)
    return topic_model, topics, topic_words


# ── 5. LLM Agent ─────────────────────────────────────────────────────────────

class LLMAgent:
    """
    Flan-T5 agent for propose / critique / refine.
    Prompts are tightly constrained to produce SHORT labels (3-5 words).
    """

    def __init__(self, model_name: str = LLM_MODEL):
        print(f"Loading LLM: {model_name}", flush=True)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

    def _generate(self, prompt: str, max_new_tokens: int = 12) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=400,
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        result = self.tokenizer.decode(
            output[0], skip_special_tokens=True
        ).strip()
        # Take only the first line and enforce 3-5 word limit
        result = result.split("\n")[0].strip(" .,:-")
        words = result.split()
        if len(words) > 5:
            result = " ".join(words[:5])
        return result if result else "general product topic"

    def propose_label(self, keywords: list, reviews: list) -> str:
        """
        Produce a SHORT noun-phrase label (3-5 words).
        Flan-T5 needs very direct, constrained prompts.
        """
        kw_str = ", ".join(keywords[:6])
        # Use only the first sentence of the first review
        rv_short = reviews[0].split(".")[0][:100] if reviews else ""
        prompt = (
            f"Write a 3-word noun phrase describing a product category.\n"
            f"Do not use verbs, brand names, or commands.\n"
            f"Keywords: {kw_str}\n"
            f"Review: {rv_short}\n"
            f"Category:"
        )
        return self._generate(prompt, max_new_tokens=10)

    def critique_label(self, label: str, keywords: list,
                       reviews: list, existing_labels: list,
                       embedder) -> dict:
        """
        Score via embedding similarity between label and its evidence.
        More reliable than yes/no LLM scoring for short labels.
        """
        label_emb = embedder.encode(
            [label], normalize_embeddings=True
        )
        review_embs = embedder.encode(
            reviews[:10], normalize_embeddings=True
        )
        sim_scores = cosine_similarity(label_emb, review_embs)[0]
        evidence_score = float(sim_scores.mean())

        # Check distinctiveness against existing labels
        if existing_labels:
            exist_embs = embedder.encode(
                existing_labels, normalize_embeddings=True
            )
            overlap_sims = cosine_similarity(label_emb, exist_embs)[0]
            distinctiveness = 1.0 - float(overlap_sims.max())
        else:
            distinctiveness = 1.0

        # Combined score
        score = round((evidence_score * 0.6) + (distinctiveness * 0.4), 3)
        feedback = (
            f"evidence_sim={evidence_score:.2f}, "
            f"distinctiveness={distinctiveness:.2f}"
        )
        return {"score": score, "feedback": feedback}

    def refine_label(self, old_label: str, feedback: str,
                     keywords: list, reviews: list) -> str:
        kw_str = ", ".join(keywords[:5])
        rv_short = reviews[0].split(".")[0][:80] if reviews else ""
        prompt = (
            f"Give a better 3-word topic label.\n"
            f"Bad label: {old_label}\n"
            f"Keywords: {kw_str}\n"
            f"Review: {rv_short}\n"
            f"Better label:"
        )
        return self._generate(prompt, max_new_tokens=10)


# ── 6. Agentic refinement loop ────────────────────────────────────────────────

def agentic_loop(topic_words: dict, docs: list,
                 embedder, index,
                 agent: LLMAgent,
                 max_iter: int = MAX_ITER) -> list:
    """
    Core contribution:
    For each topic: retrieve evidence → propose label →
    critique (embedding-based) → refine if weak → deduplicate.
    Repeat for max_iter iterations.
    """
    # Initialise states
    states = []
    for tid, words in topic_words.items():
        query = " ".join(words[:5])
        evidence = retrieve(query, embedder, index, docs, TOP_K_RETRIEVE)
        states.append({
            "topic_id": tid,
            "keywords": words,
            "label": "",
            "score": 0.0,
            "feedback": "",
            "evidence": evidence,
            "history": [],
        })

    for iteration in range(max_iter):
        print(f"\n── Agentic iteration {iteration + 1}/{max_iter} ──",
              flush=True)
        existing_labels = [s["label"] for s in states if s["label"]]

        for s in states:
            # PROPOSE
            s["label"] = agent.propose_label(
                s["keywords"], s["evidence"]
            )

            # CRITIQUE (embedding-based scoring)
            critique = agent.critique_label(
                s["label"], s["keywords"],
                s["evidence"], existing_labels, embedder,
            )
            s["score"] = critique["score"]
            s["feedback"] = critique["feedback"]

            # REFINE if score below threshold
            if s["score"] < 0.45:
                s["label"] = agent.refine_label(
                    s["label"], s["feedback"],
                    s["keywords"], s["evidence"],
                )

            s["history"].append({
                "iter": iteration + 1,
                "label": s["label"],
                "score": round(s["score"], 3),
            })
            print(
                f"  Topic {s['topic_id']:>2}: "
                f"'{s['label']:<35}' | score={s['score']:.3f}",
                flush=True,
            )

        # Deduplicate overlapping topics after each iteration
        states = deduplicate_topics(states, embedder)

    return states


def deduplicate_topics(states: list, embedder) -> list:
    if len(states) < 2:
        return states
    labels = [s["label"] for s in states]
    embs = embedder.encode(labels, normalize_embeddings=True)
    sim = cosine_similarity(embs)
    keep = [True] * len(states)
    for i in range(len(states)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(states)):
            if not keep[j]:
                continue
            if sim[i, j] > OVERLAP_SIM:
                drop = j if states[i]["score"] >= states[j]["score"] else i
                keep[drop] = False
    result = [s for s, k in zip(states, keep) if k]
    removed = len(states) - len(result)
    if removed:
        print(f"  Deduplicated: removed {removed} overlapping topic(s).",
              flush=True)
    return result


# ── 7. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(states: list, docs: list, embedder) -> dict:
    print("\nRunning evaluation...", flush=True)

    tokenized = [d.lower().split() for d in docs]
    dictionary = Dictionary(tokenized)

    # FIX: filter keywords to only words that exist in the corpus dictionary
    # This prevents npmi = nan caused by out-of-vocabulary words
    valid_words = set(dictionary.token2id.keys())
    topic_words_list = []
    for s in states:
        filtered = [w for w in s["keywords"] if w in valid_words]
        # Need at least 2 words for coherence calculation
        if len(filtered) >= 2:
            topic_words_list.append(filtered)

    if len(topic_words_list) < 2:
        print("Warning: too few valid topics for coherence.", flush=True)
        cv, npmi = 0.0, 0.0
    else:
        cv = CoherenceModel(
            topics=topic_words_list,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence()

        npmi = CoherenceModel(
            topics=topic_words_list,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
        ).get_coherence()

    # Topic diversity
    all_words = [w for t in topic_words_list for w in t[:25]]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0.0

    # Semantic separation between topic label embeddings
    topic_labels = [s["label"] for s in states]
    label_embs = embedder.encode(topic_labels, normalize_embeddings=True)
    sim_mat = cosine_similarity(label_embs)
    n = len(topic_labels)
    if n > 1:
        off_diag = (sim_mat.sum() - np.trace(sim_mat)) / (n * (n - 1))
        semantic_sep = round(1.0 - float(off_diag), 4)
    else:
        semantic_sep = 1.0

    # Semantic validity
    rev_embs = embedder.encode(
        docs[:200], normalize_embeddings=True
    )
    valid = sum(
        1 for t_emb in label_embs
        if cosine_similarity([t_emb], rev_embs)[0].max() >= 0.30
    )
    semantic_validity = round(valid / len(label_embs), 4)

    return {
        "n_topics": len(states),
        "cv_coherence": round(float(cv), 4),
        "npmi_coherence": round(float(npmi), 4),
        "topic_diversity": round(diversity, 4),
        "semantic_separation": semantic_sep,
        "semantic_validity": semantic_validity,
    }


# ── 8. Save outputs ───────────────────────────────────────────────────────────

def save_results(states: list, metrics: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Final topics text
    lines = []
    for i, s in enumerate(states, 1):
        lines.append(f"Topic {i}: {s['label']}")
        lines.append(f"  Keywords : {', '.join(s['keywords'][:8])}")
        lines.append(f"  Score    : {s['score']:.3f}")
        lines.append(f"  Feedback : {s['feedback']}")
        lines.append(f"  Sample   : {s['evidence'][0][:200] if s['evidence'] else 'N/A'}")
        lines.append("")
    (OUTPUT_DIR / "final_topics.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Full evaluation JSON
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "topics": [
                {
                    "topic_id": s["topic_id"],
                    "label": s["label"],
                    "keywords": s["keywords"][:10],
                    "score": round(s["score"], 4),
                    "history": s["history"],
                    "representative_review": (
                        s["evidence"][0][:300] if s["evidence"] else ""
                    ),
                }
                for s in states
            ],
        }, f, indent=2)

    # Human evaluation sheet
    hlines = [
        "HUMAN EVALUATION SHEET — ARTM Topics",
        "=" * 50, "",
        "Rate each topic 1–5 on three dimensions:",
        "  Relevance     : Does the label match the reviews?",
        "  Clarity       : Is the label easy to understand?",
        "  Distinctiveness: Is it different from other topics?", "",
        "Use 3 independent annotators.",
        "Compute Krippendorff's alpha for inter-annotator agreement.", "",
    ]
    for i, s in enumerate(states, 1):
        hlines += [
            f"Topic {i}: {s['label']}",
            f"  Keywords: {', '.join(s['keywords'][:6])}",
            f"  Sample: {s['evidence'][0][:200] if s['evidence'] else 'N/A'}",
            "  Annotator 1 — Relevance:___ Clarity:___ Distinctiveness:___",
            "  Annotator 2 — Relevance:___ Clarity:___ Distinctiveness:___",
            "  Annotator 3 — Relevance:___ Clarity:___ Distinctiveness:___",
            "",
        ]
    (OUTPUT_DIR / "human_eval_sheet.txt").write_text(
        "\n".join(hlines), encoding="utf-8"
    )
    print(f"\nAll outputs saved to {OUTPUT_DIR}/", flush=True)


def print_summary(states: list, metrics: dict):
    print("\n" + "=" * 60)
    print("FINAL TOPICS")
    print("=" * 60)
    for i, s in enumerate(states, 1):
        print(f"  {i:>2}. {s['label']:<40} score={s['score']:.3f}")
    print("\n" + "=" * 60)
    print("EVALUATION METRICS  (use these in your paper Table 1)")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")


# ── 9. Main ───────────────────────────────────────────────────────────────────

def main():
    np.random.seed(RANDOM_SEED)
    print("=" * 60)
    print("ARTM Pipeline v3 — Amazon Musical Instruments")
    print("=" * 60, flush=True)

    # 1. Load ALL reviews
    raw_docs = load_reviews(DATA_FILE)
    clean_docs = [clean_text(r) for r in raw_docs]
    print(f"After cleaning: {len(clean_docs):,} documents ready.", flush=True)

    # 2. Embed + FAISS index
    embedder, embs = build_embeddings(clean_docs)
    index = build_faiss_index(embs)

    # 3. BERTopic seed topics
    _, _, topic_words = run_bertopic(
        clean_docs, embs, embedder, NUM_TOPICS
    )

    # 4. LLM Agent (MPS on M4)
    agent = LLMAgent(LLM_MODEL)

    # 5. Agentic refinement loop
    final_states = agentic_loop(
        topic_words, clean_docs, embedder, index, agent, MAX_ITER
    )

    # 6. Evaluate
    metrics = evaluate(final_states, clean_docs, embedder)

    # 7. Save + print
    save_results(final_states, metrics)
    print_summary(final_states, metrics)
    print("\nDone! Check outputs_v3/ folder for all result files.", flush=True)


if __name__ == "__main__":
    main()

def run_baseline():
    """BERTopic alone — no agentic loop. Paper baseline."""
    print("\n" + "="*60)
    print("RUNNING BASELINE (BERTopic only — no agent)")
    print("="*60, flush=True)
    raw_docs = load_reviews(DATA_FILE)
    clean_docs = [clean_text(r) for r in raw_docs]
    embedder, embs = build_embeddings(clean_docs)
    _, _, topic_words = run_bertopic(clean_docs, embs, embedder, NUM_TOPICS)

    tokenized = [d.lower().split() for d in clean_docs]
    dictionary = Dictionary(tokenized)
    valid_words = set(dictionary.token2id.keys())
    topic_words_list = [
        [w for w in words if w in valid_words]
        for words in topic_words.values()
        if len([w for w in words if w in valid_words]) >= 2
    ]
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

    print(f"  n_topics       : {len(topic_words_list)}")
    print(f"  cv_coherence   : {round(float(cv), 4)}")
    print(f"  npmi_coherence : {round(float(npmi), 4)}")
    print(f"  topic_diversity: {round(diversity, 4)}")
    print("\nBaseline topic keywords:")
    for i, words in enumerate(topic_words_list, 1):
        print(f"  {i}. {', '.join(words[:6])}")