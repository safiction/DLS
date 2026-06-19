import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Callable
import ir_datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load data from BEIR nfcorpus/test
print("Loading BEIR nfcorpus/test dataset...")
ds = ir_datasets.load("beir/nfcorpus/test")
docs = {d.doc_id: d.text for d in ds.docs_iter()}
queries = {q.query_id: q.text for q in ds.queries_iter()}
qrels = {}
for qr in ds.qrels_iter():
    qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

print(f"Loaded {len(docs)} documents, {len(queries)} queries, {len(qrels)} with judgments")

def tokenize_basic(text: str) -> List[str]:
    lowercased = text.lower()
    stripped = re.sub(r'[^\w\s]', '', lowercased)
    return stripped.split()

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
}


def tokenize_stopwords(text: str) -> List[str]:
    tokens = tokenize_basic(text)
    return [t for t in tokens if t not in STOPWORDS]


def porter_stem(word: str) -> str:
    suffixes = [
        ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'), ('anci', 'ance'),
        ('izer', 'ize'), ('bli', 'ble'), ('alli', 'al'), ('entli', 'ent'),
        ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
        ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
        ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'), ('biliti', 'ble'),
        ('ing', ''), ('ly', ''), ('ed', ''), ('ies', 'y'), ('ied', 'y'),
        ('s', ''), ('es', '')
    ]
    
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)] + replacement
    return word


def tokenize_stemming(text: str) -> List[str]:
    tokens = tokenize_basic(text)
    return [porter_stem(t) for t in tokens]

def build_inverted_index(corpus: Dict[str, str], tokenizer: Callable) -> Tuple[Dict, Dict, Dict, float, int]:
    index = defaultdict(list)
    doc_term_freqs = {}
    doc_lengths = {}
    
    for doc_id, text in corpus.items():
        tokens = tokenizer(text)
        doc_lengths[doc_id] = len(tokens)
        
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        
        doc_term_freqs[doc_id] = dict(term_freqs)
    
    for doc_id, term_freqs in doc_term_freqs.items():
        for term, tf in term_freqs.items():
            index[term].append((doc_id, tf))
    
    N = len(corpus)
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 0
    
    return dict(index), doc_term_freqs, doc_lengths, avgdl, N

def compute_df(index: Dict, term: str) -> int:
    return len(index.get(term, []))


def rank_boolean_or(query_tokens: List[str], corpus: Dict, index: Dict, doc_term_freqs: Dict) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    
    for term in query_tokens:
        if term in index:
            for doc_id, _ in index[term]:
                scores[doc_id] = 1.0
    
    for doc_id in corpus:
        if doc_id not in scores:
            scores[doc_id] = 0.0
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_tfidf(query_tokens: List[str], corpus: Dict, index: Dict, doc_term_freqs: Dict, N: int) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    doc_norms = defaultdict(float)
    
    query_idf = {}
    for term in set(query_tokens):
        df = compute_df(index, term)
        if df > 0:
            query_idf[term] = math.log(N / df)
    
    for term in query_tokens:
        if term not in index:
            continue
        idf = query_idf.get(term, 0)
        for doc_id, tf in index[term]:
            weight = tf * idf
            scores[doc_id] += weight
            doc_norms[doc_id] += weight ** 2
    
    for doc_id in scores:
        norm = math.sqrt(doc_norms[doc_id])
        if norm > 0:
            scores[doc_id] /= norm
    
    for doc_id in corpus:
        if doc_id not in scores:
            scores[doc_id] = 0.0
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_bm25(query_tokens: List[str], corpus: Dict, index: Dict, doc_term_freqs: Dict, 
              doc_lengths: Dict, avgdl: float, N: int, k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    
    for term in query_tokens:
        if term not in index:
            continue
        
        df = len(index[term])
        if df == 0:
            continue
        
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        
        for doc_id, tf in index[term]:
            doc_len = doc_lengths.get(doc_id, 0)
            denominator = k1 * ((1 - b) + b * (doc_len / avgdl)) + tf
            B = ((k1 + 1) * tf) / denominator
            scores[doc_id] += idf * B
    
    for doc_id in corpus:
        if doc_id not in scores:
            scores[doc_id] = 0.0
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_rrf(tfidf_ranking: List[Tuple[str, float]], bm25_ranking: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
    scores = defaultdict(float)
    
    for rank, (doc_id, _) in enumerate(tfidf_ranking, start=1):
        scores[doc_id] += 1.0 / (k + rank)
    
    for rank, (doc_id, _) in enumerate(bm25_ranking, start=1):
        scores[doc_id] += 1.0 / (k + rank)
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def compute_dcg(relevances: List[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg_at_k(retrieved: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    relevances = [qrels.get(doc_id, 0) for doc_id in retrieved[:k]]
    dcg = compute_dcg(relevances)
    
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    idcg = compute_dcg(ideal_relevances)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_ap(retrieved: List[str], qrels: Dict[str, int]) -> float:
    relevant_docs = set(doc_id for doc_id, rel in qrels.items() if rel > 0)
    if not relevant_docs:
        return 0.0
    
    num_relevant = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_docs:
            num_relevant += 1
            precision_sum += num_relevant / i
    
    return precision_sum / len(relevant_docs)


def compute_mrr(retrieved: List[str], qrels: Dict[str, int]) -> float:
    relevant_docs = set(doc_id for doc_id, rel in qrels.items() if rel > 0)
    
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_docs:
            return 1.0 / i
    return 0.0


def evaluate_ranker(rankings: Dict[str, List[Tuple[str, float]]], qrels: Dict) -> Dict[str, float]:
    ndcg_scores = []
    map_scores = []
    mrr_scores = []
    
    for query_id, ranking in rankings.items():
        if query_id not in qrels:
            continue
        
        retrieved = [doc_id for doc_id, _ in ranking]
        query_qrels = qrels[query_id]
        
        ndcg_scores.append(compute_ndcg_at_k(retrieved, query_qrels))
        map_scores.append(compute_ap(retrieved, query_qrels))
        mrr_scores.append(compute_mrr(retrieved, query_qrels))
    
    return {
        'nDCG@10': np.mean(ndcg_scores),
        'MAP': np.mean(map_scores),
        'MRR': np.mean(mrr_scores),
        'nDCG_per_query': ndcg_scores,
        'MAP_per_query': map_scores,
        'MRR_per_query': mrr_scores
    }


def run_main_experiment():
    print("\nMain Experiment: Comparing Rankers")
    print("-" * 50)
    
    print("\nBuilding inverted index...")
    index, doc_term_freqs, doc_lengths, avgdl, N = build_inverted_index(docs, tokenize_basic)
    print(f"Vocabulary size: {len(index)} terms, avgdl: {avgdl:.1f}")
    
    query_ids = sorted([qid for qid in queries.keys() if qid in qrels])
    print(f"Evaluating on {len(query_ids)} queries with judgments")
    
    print("\nRunning rankers...")
    
    boolean_rankings = {}
    tfidf_rankings = {}
    bm25_rankings = {}
    rrf_rankings = {}
    
    for query_id in query_ids:
        query_text = queries[query_id]
        query_tokens = tokenize_basic(query_text)
        
        if not query_tokens:
            continue
        
        boolean_rankings[query_id] = rank_boolean_or(query_tokens, docs, index, doc_term_freqs)
        tfidf_rankings[query_id] = rank_tfidf(query_tokens, docs, index, doc_term_freqs, N)
        bm25_rankings[query_id] = rank_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N)
        rrf_rankings[query_id] = rank_rrf(tfidf_rankings[query_id], bm25_rankings[query_id])
    
    print("\nEvaluating metrics...")
    boolean_results = evaluate_ranker(boolean_rankings, qrels)
    tfidf_results = evaluate_ranker(tfidf_rankings, qrels)
    bm25_results = evaluate_ranker(bm25_rankings, qrels)
    rrf_results = evaluate_ranker(rrf_rankings, qrels)
    
    print("\nResults: Mean Metrics Across All Queries")
    print("-" * 50)
    print(f"{'Ranker':<15} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 50)
    print(f"{'Boolean-OR':<15} {boolean_results['nDCG@10']:>10.4f} {boolean_results['MAP']:>10.4f} {boolean_results['MRR']:>10.4f}")
    print(f"{'TF-IDF':<15} {tfidf_results['nDCG@10']:>10.4f} {tfidf_results['MAP']:>10.4f} {tfidf_results['MRR']:>10.4f}")
    print(f"{'BM25':<15} {bm25_results['nDCG@10']:>10.4f} {bm25_results['MAP']:>10.4f} {bm25_results['MRR']:>10.4f}")
    print(f"{'RRF Fusion':<15} {rrf_results['nDCG@10']:>10.4f} {rrf_results['MAP']:>10.4f} {rrf_results['MRR']:>10.4f}")
    
    return tfidf_results, bm25_results


def run_bm25_parameter_sweep():
    print("\n\nBM25 Parameter Sweep")
    print("-" * 50)
    
    index, doc_term_freqs, doc_lengths, avgdl, N = build_inverted_index(docs, tokenize_basic)
    query_ids = sorted([qid for qid in queries.keys() if qid in qrels])
    
    k1_values = [0.5, 1.0, 1.2, 1.5, 2.0, 3.0]
    b_values = [0, 0.25, 0.5, 0.75, 1.0]
    
    print("\nSweeping k1 (with b=0.75)")
    k1_scores = []
    for k1 in k1_values:
        rankings = {}
        for query_id in query_ids:
            query_tokens = tokenize_basic(queries[query_id])
            if query_tokens:
                rankings[query_id] = rank_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N, k1=k1, b=0.75)
        
        results = evaluate_ranker(rankings, qrels)
        k1_scores.append(results['nDCG@10'])
        print(f"  k1={k1}: nDCG@10={results['nDCG@10']:.4f}")
    
    print("\nSweeping b (with k1=1.5)")
    b_scores = []
    for b in b_values:
        rankings = {}
        for query_id in query_ids:
            query_tokens = tokenize_basic(queries[query_id])
            if query_tokens:
                rankings[query_id] = rank_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N, k1=1.5, b=b)
        
        results = evaluate_ranker(rankings, qrels)
        b_scores.append(results['nDCG@10'])
        print(f"  b={b}: nDCG@10={results['nDCG@10']:.4f}")
    
    print("\nSearching for best (k1, b) combination")
    best_score = 0
    best_params = None
    for k1 in k1_values:
        for b in b_values:
            rankings = {}
            for query_id in query_ids:
                query_tokens = tokenize_basic(queries[query_id])
                if query_tokens:
                    rankings[query_id] = rank_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N, k1=k1, b=b)
            
            results = evaluate_ranker(rankings, qrels)
            if results['nDCG@10'] > best_score:
                best_score = results['nDCG@10']
                best_params = (k1, b)
    
    print(f"Best parameters: k1={best_params[0]}, b={best_params[1]} with nDCG@10={best_score:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k1_values, k1_scores, 'o-', color='steelblue')
    ax1.set_xlabel('k1 (saturation parameter)')
    ax1.set_ylabel('nDCG@10')
    ax1.set_title('BM25: nDCG@10 vs k1 (b=0.75)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(b_values, b_scores, 'o-', color='steelblue')
    ax2.set_xlabel('b (length normalization)')
    ax2.set_ylabel('nDCG@10')
    ax2.set_title('BM25: nDCG@10 vs b (k1=1.5)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bm25_parameter_sweep.png', dpi=150)
    print("Plot saved to bm25_parameter_sweep.png")
    
    return k1_values, k1_scores, b_values, b_scores


def run_tokenization_study():
    print("\n\nTokenization Study")
    print("-" * 50)
    
    tokenizers = {
        'basic': tokenize_basic,
        'stopwords': tokenize_stopwords,
        'stemming': tokenize_stemming
    }
    
    query_ids = sorted([qid for qid in queries.keys() if qid in qrels])
    
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"\nTesting {name} tokenization")
        
        index, doc_term_freqs, doc_lengths, avgdl, N = build_inverted_index(docs, tokenizer)
        print(f"  Vocabulary size: {len(index)} terms")
        
        rankings = {}
        for query_id in query_ids:
            query_tokens = tokenizer(queries[query_id])
            if query_tokens:
                rankings[query_id] = rank_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N)
        
        eval_results = evaluate_ranker(rankings, qrels)
        results[name] = eval_results
        
        print(f"  nDCG@10: {eval_results['nDCG@10']:.4f}")
        print(f"  MAP: {eval_results['MAP']:.4f}")
        print(f"  MRR: {eval_results['MRR']:.4f}")
    
    print("\nTokenization Comparison")
    print("-" * 50)
    print(f"{'Tokenizer':<15} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 50)
    for name in tokenizers.keys():
        print(f"{name:<15} {results[name]['nDCG@10']:>10.4f} {results[name]['MAP']:>10.4f} {results[name]['MRR']:>10.4f}")
    
    return results


def run_significance_test(tfidf_results, bm25_results):
    print("\n\nStatistical Significance Testing")
    print("-" * 50)
    
    tfidf_ndcg = tfidf_results['nDCG_per_query']
    bm25_ndcg = bm25_results['nDCG_per_query']
    
    t_stat, t_pvalue = stats.ttest_rel(bm25_ndcg, tfidf_ndcg)
    w_stat, w_pvalue = stats.wilcoxon(bm25_ndcg, tfidf_ndcg)
    
    # Mean difference and 95% CI
    differences = [b - t for b, t in zip(bm25_ndcg, tfidf_ndcg)]
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    ci_lower = mean_diff - 1.96 * std_diff / math.sqrt(n)
    ci_upper = mean_diff + 1.96 * std_diff / math.sqrt(n)
    
    print(f"\nComparing BM25 vs TF-IDF on nDCG@10:")
    print(f"  Mean difference (BM25 - TF-IDF): {mean_diff:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {t_pvalue:.4f}")
    print(f"  Significant at alpha=0.05: {'Yes' if t_pvalue < 0.05 else 'No'}")
    
    print(f"\nWilcoxon signed-rank test:")
    print(f"  statistic: {w_stat:.4f}")
    print(f"  p-value: {w_pvalue:.4f}")
    print(f"  Significant at alpha=0.05: {'Yes' if w_pvalue < 0.05 else 'No'}")
    
    print("\nInterpretation:")
    if t_pvalue < 0.05:
        print("  The improvement of BM25 over TF-IDF is statistically significant.")
    else:
        print("  The difference between BM25 and TF-IDF is not statistically significant.")

if __name__ == "__main__":
    tfidf_results, bm25_results = run_main_experiment()
    
    run_bm25_parameter_sweep()
    
    run_tokenization_study()
    
    run_significance_test(tfidf_results, bm25_results)