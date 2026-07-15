import math
import re
import numpy as np
import ir_datasets
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import wilcoxon, ttest_rel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import warnings
import os

warnings.filterwarnings("ignore")

# ==========================================
# 1. DATA LOADING & EVALUATION METRICS
# ==========================================

def load_nfcorpus():
    ds = ir_datasets.load("beir/nfcorpus/test")
    corpus = {doc.doc_id: {"title": "", "text": doc.text} for doc in ds.docs_iter()}
    queries = [{"_id": q.query_id, "text": q.text} for q in ds.queries_iter()]
    qrels = defaultdict(dict)
    for qr in ds.qrels_iter():
        qrels[qr.query_id][qr.doc_id] = qr.relevance
    return corpus, queries, qrels

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def dcg(rels):
    return sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(rels, 1))

def ndcg_at_k(ranking, qrels, k=10):
    rels = [qrels.get(doc_id, 0) for doc_id, _ in ranking[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    if not ideal or dcg(ideal) == 0:
        return 0.0
    return dcg(rels) / dcg(ideal)

def average_precision(ranking, qrels):
    hits = 0
    sum_precisions = 0.0
    relevant_docs = sum(1 for v in qrels.values() if v > 0)
    if relevant_docs == 0:
        return 0.0
    for i, (doc_id, _) in enumerate(ranking):
        if qrels.get(doc_id, 0) > 0:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / relevant_docs

def reciprocal_rank(ranking, qrels):
    for i, (doc_id, _) in enumerate(ranking):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

# ==========================================
# 2. RETRIEVERS
# ==========================================

class BM25Scout:
    def __init__(self, corpus):
        self.index = defaultdict(list)
        self.doc_lengths = {}
        self.doc_tf = {}
        self.corpus = corpus
        self.N = len(corpus)

        for doc_id, doc in corpus.items():
            text = doc["title"] + " " + doc["text"]
            tokens = tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.doc_tf[doc_id] = tf

        for doc_id, tf in self.doc_tf.items():
            for term, freq in tf.items():
                self.index[term].append((doc_id, freq))

        self.avgdl = sum(self.doc_lengths.values()) / max(1, len(self.doc_lengths))

    def retrieve(self, query, top_k=100, k1=1.5, b=0.75):
        tokens = tokenize(query)
        scores = defaultdict(float)
        for term in tokens:
            postings = self.index.get(term, [])
            df = len(postings)
            if df == 0:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings:
                numerator = (k1 + 1) * tf
                denominator = k1 * ((1 - b) + b * self.doc_lengths[doc_id] / self.avgdl) + tf
                scores[doc_id] += idf * numerator / denominator
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

class DenseScout:
    def __init__(self, corpus):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.ids = list(corpus.keys())
        texts = [corpus[i]["text"] for i in self.ids]
        emb = self.model.encode(texts, show_progress_bar=False)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def retrieve(self, query, top_k=100):
        q = self.model.encode([query])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        scores = (self.emb @ q.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in idx]

class SpladeScout:
    def __init__(self, corpus):
        self.tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.model.eval()
        self.ids = list(corpus.keys())
        # Pre-compute document sparse vectors
        self.doc_vecs = []
        for doc_id in self.ids:
            vec = self._get_sparse_vector(corpus[doc_id]["text"])
            # Keep only non-zero entries as dict for fast dot product
            nz = torch.nonzero(vec, as_tuple=False).squeeze(1)
            vec_dict = {int(i): float(vec[i]) for i in nz}
            self.doc_vecs.append(vec_dict)

    def _get_sparse_vector(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        weights = torch.log(1 + torch.relu(logits[0]))
        sparse_vec = torch.max(weights, dim=0).values
        return sparse_vec

    def retrieve(self, query, top_k=100):
        q_vec = self._get_sparse_vector(query)
        nz = torch.nonzero(q_vec, as_tuple=False).squeeze(1)
        q_dict = {int(i): float(q_vec[i]) for i in nz}
        scores = []
        for i, doc_vec in enumerate(self.doc_vecs):
            score = 0.0
            for tok_id, qw in q_dict.items():
                score += qw * doc_vec.get(tok_id, 0.0)
            scores.append((self.ids[i], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ==========================================
# 3. HYBRID FUSION
# ==========================================

def min_max_norm(scores_dict):
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    min_v, max_v = min(vals), max(vals)
    if max_v == min_v:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

def z_score_norm(scores_dict):
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mu, std = np.mean(vals), np.std(vals)
    if std == 0:
        return {k: 0.0 for k in scores_dict}
    return {k: (v - mu) / std for k, v in scores_dict.items()}

def rrf(rankings_list, k=60):
    fused = defaultdict(float)
    for ranking in rankings_list:
        for rank, (doc_id, _) in enumerate(ranking):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def convex_combination(ranking1, ranking2, alpha=0.5, norm_fn=z_score_norm):
    dict1 = norm_fn(dict(ranking1))
    dict2 = norm_fn(dict(ranking2))
    all_docs = set(dict1.keys()) | set(dict2.keys())
    fused = {}
    for d in all_docs:
        s1 = dict1.get(d, 0)
        s2 = dict2.get(d, 0)
        fused[d] = alpha * s1 + (1 - alpha) * s2
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def normalized_score_fusion(rankings, method='minmax'):
    all_docs = set()
    dicts = []
    for ranking in rankings:
        d = {doc_id: score for doc_id, score in ranking}
        dicts.append(d)
        all_docs.update(d.keys())
    normalized = []
    for d in dicts:
        if method == 'minmax':
            normalized.append(min_max_norm(d))
        else:
            normalized.append(z_score_norm(d))
    fused = {}
    for doc_id in all_docs:
        fused[doc_id] = sum(nd.get(doc_id, 0.0) for nd in normalized)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

# ==========================================
# 4. LEARNING TO RANK (RankNet & LambdaRank)
# ==========================================

class LinearCombiner(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.w = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return self.w(x)

def build_features(query, qid, bm25_pool, dense_sys, splade_sys, bm25_sys):
    """Build per-(query, doc) feature vector from pre-computed retrievals."""
    qtext = query["text"]

    # Get scores for all docs in the union pool
    bm25_scores = dict(bm25_pool)
    dense_scores = dict(dense_sys.retrieve(qtext, top_k=2000))
    splade_scores = dict(splade_sys.retrieve(qtext, top_k=2000))

    # Union of top-100 from all retrievers
    all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys()) | set(splade_scores.keys())

    # Normalize scores
    b_norm = z_score_norm(bm25_scores)
    d_norm = z_score_norm(dense_scores)
    s_norm = z_score_norm(splade_scores)

    features = []
    labels = []
    doc_ids = []

    for doc_id in all_doc_ids:
        dl = bm25_sys.doc_lengths.get(doc_id, 0)
        dl_norm = (dl - bm25_sys.avgdl) / bm25_sys.avgdl if bm25_sys.avgdl > 0 else 0.0

        f_vec = [
            b_norm.get(doc_id, 0),
            d_norm.get(doc_id, 0),
            s_norm.get(doc_id, 0),
            dl_norm,
        ]
        rel = qrels.get(qid, {}).get(doc_id, 0)
        features.append(f_vec)
        labels.append(rel)
        doc_ids.append(doc_id)

    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32), doc_ids

def train_ltr(features_list, labels_list, use_lambda=False, epochs=50, lr=0.01):
    model = LinearCombiner(num_features=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for X, Y in zip(features_list, labels_list):
            if len(X) < 2:
                continue
            optimizer.zero_grad()
            scores = model(X).squeeze()

            idx_i, idx_j = torch.combinations(torch.arange(len(Y)), r=2).unbind(1)
            rel_diff = Y[idx_i] - Y[idx_j]
            valid_pairs = rel_diff != 0

            if not valid_pairs.any():
                continue

            i_valid = idx_i[valid_pairs]
            j_valid = idx_j[valid_pairs]
            rel_diff_valid = rel_diff[valid_pairs]

            swap = rel_diff_valid < 0
            i_final = torch.where(swap, j_valid, i_valid)
            j_final = torch.where(swap, i_valid, j_valid)

            s_i = scores[i_final]
            s_j = scores[j_final]

            loss_ij = torch.log(1 + torch.exp(-(s_i - s_j)))

            if use_lambda:
                lambda_weights = torch.abs(Y[i_final] - Y[j_final])
                loss_ij = loss_ij * lambda_weights

            loss = loss_ij.mean()
            loss.backward()
            optimizer.step()

    return model

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    os.makedirs("lab_02", exist_ok=True)

    print("Loading NFCorpus...")
    corpus, queries, qrels = load_nfcorpus()

    # 30 queries for balance between speed and statistics
    test_queries = sorted(queries, key=lambda x: x["_id"])[:30]
    # Split: 20 train, 10 validation
    train_queries = test_queries[:20]
    val_queries = test_queries[20:]

    print("Initializing Scouts...")
    bm25 = BM25Scout(corpus)
    dense = DenseScout(corpus)
    splade = SpladeScout(corpus)

    # Pre-compute retrievals for all queries
    print("Pre-computing retrievals...")
    all_retrievals = {}
    for q in test_queries:
        qid = q["_id"]
        qtext = q["text"]
        all_retrievals[qid] = {
            "bm25": bm25.retrieve(qtext, top_k=100),
            "dense": dense.retrieve(qtext, top_k=100),
            "splade": splade.retrieve(qtext, top_k=100),
        }

    # ------------------------------------------------------------------
    # Evaluate individual retrievers
    # ------------------------------------------------------------------
    systems = {"bm25": [], "dense": [], "splade": []}
    for q in test_queries:
        qid = q["_id"]
        for sys_name in systems:
            ranking = all_retrievals[qid][sys_name]
            systems[sys_name].append({
                "ndcg": ndcg_at_k(ranking, qrels.get(qid, {})),
                "map": average_precision(ranking, qrels.get(qid, {})),
                "mrr": reciprocal_rank(ranking, qrels.get(qid, {})),
            })

    # ------------------------------------------------------------------
    # Fusion experiments
    # ------------------------------------------------------------------
    fusion_results = {
        "rrf": [],
        "convex_0.0": [],
        "convex_0.25": [],
        "convex_0.5": [],
        "convex_0.75": [],
        "convex_1.0": [],
        "minmax": [],
        "zscore": [],
    }

    for q in test_queries:
        qid = q["_id"]
        bm25_r = all_retrievals[qid]["bm25"]
        dense_r = all_retrievals[qid]["dense"]
        splade_r = all_retrievals[qid]["splade"]

        # RRF
        rrf_rank = rrf([bm25_r, dense_r, splade_r], k=60)
        fusion_results["rrf"].append({
            "ndcg": ndcg_at_k(rrf_rank, qrels.get(qid, {})),
            "map": average_precision(rrf_rank, qrels.get(qid, {})),
            "mrr": reciprocal_rank(rrf_rank, qrels.get(qid, {})),
        })

        # Convex combinations dense + BM25
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            rank = convex_combination(dense_r, bm25_r, alpha=alpha, norm_fn=z_score_norm)
            fusion_results[f"convex_{alpha}"].append({
                "ndcg": ndcg_at_k(rank, qrels.get(qid, {})),
                "map": average_precision(rank, qrels.get(qid, {})),
                "mrr": reciprocal_rank(rank, qrels.get(qid, {})),
            })

        # Normalized score fusion
        minmax_rank = normalized_score_fusion([bm25_r, dense_r, splade_r], 'minmax')
        fusion_results["minmax"].append({
            "ndcg": ndcg_at_k(minmax_rank, qrels.get(qid, {})),
            "map": average_precision(minmax_rank, qrels.get(qid, {})),
            "mrr": reciprocal_rank(minmax_rank, qrels.get(qid, {})),
        })

        zscore_rank = normalized_score_fusion([bm25_r, dense_r, splade_r], 'zscore')
        fusion_results["zscore"].append({
            "ndcg": ndcg_at_k(zscore_rank, qrels.get(qid, {})),
            "map": average_precision(zscore_rank, qrels.get(qid, {})),
            "mrr": reciprocal_rank(zscore_rank, qrels.get(qid, {})),
        })

    # ------------------------------------------------------------------
    # Learning to Rank
    # ------------------------------------------------------------------
    print("Building LTR features...")
    train_features = []
    train_labels = []
    val_features = []
    val_labels = []
    val_doc_ids = []

    for q in train_queries:
        qid = q["_id"]
        X, Y, _ = build_features(q, qid, all_retrievals[qid]["bm25"], dense, splade, bm25)
        train_features.append(X)
        train_labels.append(Y)

    for q in val_queries:
        qid = q["_id"]
        X, Y, d_ids = build_features(q, qid, all_retrievals[qid]["bm25"], dense, splade, bm25)
        val_features.append(X)
        val_labels.append(Y)
        val_doc_ids.append(d_ids)

    print("Training RankNet...")
    ranknet_model = train_ltr(train_features, train_labels, use_lambda=False, epochs=50, lr=0.01)

    print("Training LambdaRank...")
    lambdarank_model = train_ltr(train_features, train_labels, use_lambda=True, epochs=50, lr=0.01)

    # Evaluate LTR on validation set
    ltr_results = {"ranknet": [], "lambdarank": []}
    for i, q in enumerate(val_queries):
        qid = q["_id"]
        X = val_features[i]
        d_ids = val_doc_ids[i]

        for model_name, model in [("ranknet", ranknet_model), ("lambdarank", lambdarank_model)]:
            model.eval()
            with torch.no_grad():
                scores = model(X).squeeze().numpy()
            ranking = sorted(zip(d_ids, scores), key=lambda x: x[1], reverse=True)
            ltr_results[model_name].append({
                "ndcg": ndcg_at_k(ranking, qrels.get(qid, {})),
                "map": average_precision(ranking, qrels.get(qid, {})),
                "mrr": reciprocal_rank(ranking, qrels.get(qid, {})),
            })

    # Best single feature baseline on validation
    best_single_val = []
    for q in val_queries:
        qid = q["_id"]
        ranking = all_retrievals[qid]["bm25"]
        best_single_val.append({
            "ndcg": ndcg_at_k(ranking, qrels.get(qid, {})),
            "map": average_precision(ranking, qrels.get(qid, {})),
            "mrr": reciprocal_rank(ranking, qrels.get(qid, {})),
        })

    # ------------------------------------------------------------------
    # Feature ablation
    # ------------------------------------------------------------------
    feature_names = ["BM25", "Dense", "SPLADE", "DocLen"]
    ablation_results = {}

    for feat_idx in range(4):
        mask = [i for i in range(4) if i != feat_idx]
        train_features_ab = [X[:, mask] for X in train_features]
        val_features_ab = [X[:, mask] for X in val_features]

        model_ab = train_ltr(train_features_ab, train_labels, use_lambda=False, epochs=50, lr=0.01)
        model_ab.eval()

        abl_scores = []
        for i, q in enumerate(val_queries):
            qid = q["_id"]
            X = val_features_ab[i]
            d_ids = val_doc_ids[i]
            with torch.no_grad():
                scores = model_ab(X).squeeze().numpy()
            ranking = sorted(zip(d_ids, scores), key=lambda x: x[1], reverse=True)
            abl_scores.append({
                "ndcg": ndcg_at_k(ranking, qrels.get(qid, {})),
                "map": average_precision(ranking, qrels.get(qid, {})),
                "mrr": reciprocal_rank(ranking, qrels.get(qid, {})),
            })
        ablation_results[feature_names[feat_idx]] = abl_scores

    # ------------------------------------------------------------------
    # Statistical significance: best system vs BM25 baseline on validation
    # ------------------------------------------------------------------
    all_val_systems = {
        "BM25": best_single_val,
        "RankNet": ltr_results["ranknet"],
        "LambdaRank": ltr_results["lambdarank"],
        "RRF": fusion_results["rrf"][20:],
        "MinMax": fusion_results["minmax"][20:],
        "ZScore": fusion_results["zscore"][20:],
    }
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        all_val_systems[f"Convex_a{alpha}"] = fusion_results[f"convex_{alpha}"][20:]

    mean_ndcg = {name: np.mean([s["ndcg"] for s in scores]) for name, scores in all_val_systems.items()}
    best_system_name = max(mean_ndcg, key=mean_ndcg.get)
    best_system_scores = [s["ndcg"] for s in all_val_systems[best_system_name]]
    baseline_scores = [s["ndcg"] for s in all_val_systems["BM25"]]

    t_stat, t_pvalue = ttest_rel(best_system_scores, baseline_scores)
    w_stat, w_pvalue = wilcoxon(best_system_scores, baseline_scores)

    differences = [b - bm for b, bm in zip(best_system_scores, baseline_scores)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    ci_lower = mean_diff - 1.96 * std_diff / math.sqrt(n)
    ci_upper = mean_diff + 1.96 * std_diff / math.sqrt(n)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("Generating plots...")

    # Plot 1: Individual retrievers comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    retriever_names = ["BM25", "Dense", "SPLADE"]
    retriever_ndcg = [np.mean([s["ndcg"] for s in systems[name]]) for name in ["bm25", "dense", "splade"]]
    ax.bar(retriever_names, retriever_ndcg, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('nDCG@10')
    ax.set_title('Individual Retriever Performance')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab_02/retrievers_comparison.png', dpi=150)
    plt.close()

    # Plot 2: Fusion methods comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    fusion_names = ['RRF', 'Convex\nα=0', 'Convex\nα=0.25', 'Convex\nα=0.5', 'Convex\nα=0.75', 'Convex\nα=1', 'MinMax', 'ZScore']
    fusion_ndcg = [
        np.mean([s["ndcg"] for s in fusion_results["rrf"]]),
        np.mean([s["ndcg"] for s in fusion_results["convex_0.0"]]),
        np.mean([s["ndcg"] for s in fusion_results["convex_0.25"]]),
        np.mean([s["ndcg"] for s in fusion_results["convex_0.5"]]),
        np.mean([s["ndcg"] for s in fusion_results["convex_0.75"]]),
        np.mean([s["ndcg"] for s in fusion_results["convex_1.0"]]),
        np.mean([s["ndcg"] for s in fusion_results["minmax"]]),
        np.mean([s["ndcg"] for s in fusion_results["zscore"]]),
    ]
    colors = ['steelblue'] * len(fusion_names)
    ax.bar(fusion_names, fusion_ndcg, color=colors)
    ax.set_ylabel('nDCG@10')
    ax.set_title('Fusion Methods Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab_02/fusion_comparison.png', dpi=150)
    plt.close()

    # Plot 3: LTR comparison on validation
    fig, ax = plt.subplots(figsize=(8, 5))
    ltr_names = ['BM25', 'RankNet', 'LambdaRank']
    ltr_ndcg = [
        np.mean([s["ndcg"] for s in best_single_val]),
        np.mean([s["ndcg"] for s in ltr_results["ranknet"]]),
        np.mean([s["ndcg"] for s in ltr_results["lambdarank"]]),
    ]
    ax.bar(ltr_names, ltr_ndcg, color=['#d62728', '#9467bd', '#8c564b'])
    ax.set_ylabel('nDCG@10')
    ax.set_title('Learning to Rank Performance (Validation Set)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab_02/ltr_comparison.png', dpi=150)
    plt.close()

    # Plot 4: Feature ablation
    fig, ax = plt.subplots(figsize=(8, 5))
    abl_names = ['Full Model'] + [f'-{f}' for f in feature_names]
    base_ltr = np.mean([s["ndcg"] for s in ltr_results["ranknet"]])
    abl_means = [base_ltr] + [np.mean([s["ndcg"] for s in ablation_results[f]]) for f in feature_names]
    colors_abl = ['green'] + ['coral'] * len(feature_names)
    ax.bar(abl_names, abl_means, color=colors_abl)
    ax.set_ylabel('nDCG@10')
    ax.set_title('Feature Ablation Study (RankNet)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab_02/feature_ablation.png', dpi=150)
    plt.close()

    # Plot 5: Per-query distribution (boxplot)
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_box = [
        [s["ndcg"] for s in systems["bm25"]],
        [s["ndcg"] for s in systems["dense"]],
        [s["ndcg"] for s in systems["splade"]],
        [s["ndcg"] for s in fusion_results["rrf"]],
    ]
    ax.boxplot(data_for_box, labels=['BM25', 'Dense', 'SPLADE', 'RRF'])
    ax.set_ylabel('nDCG@10')
    ax.set_title('Per-Query nDCG@10 Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab_02/per_query_distribution.png', dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Write results to file
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("LAB 02 RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Dataset: BEIR nfcorpus/test")
    lines.append(f"Queries: {len(test_queries)} (20 train, 10 validation)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("1. INDIVIDUAL RETRIEVERS")
    lines.append("-" * 70)
    lines.append(f"{'Retriever':<15} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    lines.append("-" * 70)
    for name, key in [("BM25", "bm25"), ("Dense", "dense"), ("SPLADE", "splade")]:
        ndcg = np.mean([s["ndcg"] for s in systems[key]])
        map_v = np.mean([s["map"] for s in systems[key]])
        mrr = np.mean([s["mrr"] for s in systems[key]])
        lines.append(f"{name:<15} {ndcg:>10.4f} {map_v:>10.4f} {mrr:>10.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("2. FUSION METHODS")
    lines.append("-" * 70)
    lines.append(f"{'Method':<25} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    lines.append("-" * 70)
    for key in ["rrf", "convex_0.0", "convex_0.25", "convex_0.5", "convex_0.75", "convex_1.0", "minmax", "zscore"]:
        ndcg = np.mean([s["ndcg"] for s in fusion_results[key]])
        map_v = np.mean([s["map"] for s in fusion_results[key]])
        mrr = np.mean([s["mrr"] for s in fusion_results[key]])
        lines.append(f"{key:<25} {ndcg:>10.4f} {map_v:>10.4f} {mrr:>10.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("3. LEARNING TO RANK (Validation Set)")
    lines.append("-" * 70)
    lines.append(f"{'Method':<20} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    lines.append("-" * 70)
    for name, key in [("BM25", None), ("RankNet", "ranknet"), ("LambdaRank", "lambdarank")]:
        if key is None:
            scores = best_single_val
        else:
            scores = ltr_results[key]
        ndcg = np.mean([s["ndcg"] for s in scores])
        map_v = np.mean([s["map"] for s in scores])
        mrr = np.mean([s["mrr"] for s in scores])
        lines.append(f"{name:<20} {ndcg:>10.4f} {map_v:>10.4f} {mrr:>10.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("4. FEATURE ABLATION (RankNet, Validation Set)")
    lines.append("-" * 70)
    lines.append(f"{'Dropped Feature':<20} {'nDCG@10':>10} {'Change':>10}")
    lines.append("-" * 70)
    base_ndcg = np.mean([s["ndcg"] for s in ltr_results["ranknet"]])
    for feat_name in feature_names:
        mean_abl = np.mean([s["ndcg"] for s in ablation_results[feat_name]])
        change = mean_abl - base_ndcg
        lines.append(f"{feat_name:<20} {mean_abl:>10.4f} {change:>+10.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("5. STATISTICAL SIGNIFICANCE")
    lines.append("-" * 70)
    lines.append(f"Best system: {best_system_name} (nDCG@10 = {mean_ndcg[best_system_name]:.4f})")
    lines.append(f"BM25 baseline: nDCG@10 = {mean_ndcg['BM25']:.4f}")
    lines.append(f"Mean difference: {mean_diff:.4f}")
    lines.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    lines.append("")
    lines.append(f"Paired t-test: t={t_stat:.4f}, p={t_pvalue:.4f}")
    lines.append(f"Wilcoxon test: W={w_stat:.4f}, p={w_pvalue:.4f}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("6. LEARNED WEIGHTS")
    lines.append("-" * 70)
    weights = ranknet_model.w.weight.squeeze().detach().numpy()
    for name, w in zip(feature_names, weights):
        lines.append(f"{name:<15}: {w:+.4f}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Plots saved:")
    lines.append("  - lab_02/retrievers_comparison.png")
    lines.append("  - lab_02/fusion_comparison.png")
    lines.append("  - lab_02/ltr_comparison.png")
    lines.append("  - lab_02/feature_ablation.png")
    lines.append("  - lab_02/per_query_distribution.png")
    lines.append("=" * 70)

    with open("lab_02/results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Done! Results written to lab_02/results.txt")
    print("Plots saved to lab_02/*.png")
