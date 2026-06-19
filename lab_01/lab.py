import math
import re  # for tokenization
from collections import defaultdict
import ir_datasets

# Load data from BEIR nfcorpus/test
ds = ir_datasets.load("beir/nfcorpus/test")
docs = {d.doc_id: d.text for d in ds.docs_iter()}
queries = {q.query_id: q.text for q in ds.queries_iter()}
qrels = {}
for qr in ds.qrels_iter():
    qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance

def tokenize(text):
    lowercased = text.lower()
    stripped = re.sub(r'[^\w\s]', '', lowercased)
    tokenized = stripped.split()
    return tokenized

def build_inverted_index(corpus):
    index = defaultdict(list)  # term -> [(doc_id, tf), ...]
    doc_term_freqs = {}  # doc_id -> {term: tf}
    doc_lengths = {}
    
    # Compute tf for each document
    for doc_id, text in corpus.items():
        tokens = tokenize(text)
        doc_lengths[doc_id] = len(tokens)
        
        # Count tf in the document
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        
        doc_term_freqs[doc_id] = term_freqs
    
    for doc_id, term_freqs in doc_term_freqs.items():
        for term, tf in term_freqs.items():
            index[term].append((doc_id, tf))
    
    # Calculate avg doc len
    N = len(corpus)
    avgdl = sum(doc_lengths.values()) / N if N > 0 else 0
    
    return dict(index), doc_term_freqs, doc_lengths, avgdl, N

def compute_df(index, term):
    if term in index:
        return len(index[term])
    return 0

def compute_idf(N, df):
    if df == 0:
        return 0
    return math.log(N / df)

def compute_tf_idf_score(query_tokens, doc_id, index, doc_term_freqs, N):
    score = 0.0
    
    for term in query_tokens:
        # Get tf in document
        tf = doc_term_freqs.get(doc_id, {}).get(term, 0)
        if tf == 0:
            continue
        
        # Compute IDF
        df = compute_df(index, term)
        idf = compute_idf(N, df)
        
        # TF-IDF weight: tf * idf
        weight = tf * idf
        score += weight
    return score

def rank_documents_tfidf(query_tokens, corpus, index, doc_term_freqs, N):
    scores = []
    for doc_id in corpus:
        score = compute_tf_idf_score(query_tokens, doc_id, index, doc_term_freqs, N)
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def compute_bm25_idf(N, df):
    if df == 0:
        return 0
    return math.log(1 + (N - df + 0.5) / (df + 0.5))

def compute_bm25_score(query_tokens, doc_id, index, doc_term_freqs, doc_lengths, avgdl, N, k1=1.5, b=0.75):
    score = 0.0
    doc_len = doc_lengths.get(doc_id, 0)
    
    for term in query_tokens:
        tf = doc_term_freqs.get(doc_id, {}).get(term, 0)
        if tf == 0:
            continue
        
        # Compute BM25 IDF
        df = compute_df(index, term)
        idf = compute_bm25_idf(N, df)
        
        # BM25 term weight with saturation and length normalization
        denominator = k1 * ((1 - b) + b * (doc_len / avgdl)) + tf
        B = ((k1 + 1) * tf) / denominator
        score += idf * B
    return score

def rank_documents_bm25(query_tokens, corpus, index, doc_term_freqs, doc_lengths, avgdl, N, k1=1.5, b=0.75):
    scores = []
    for doc_id in corpus:
        score = compute_bm25_score(query_tokens, doc_id, index, doc_term_freqs, doc_lengths, avgdl, N, k1, b)
        scores.append((doc_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def compute_dcg(relevances):
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        gain = (2 ** rel) - 1
        discount = math.log2(i + 1)
        dcg += gain / discount
    return dcg

def compute_ndcg_at_k(retrieved_docs, qrels_for_query, k=10):
    # Get relevance scores for retrieved documents
    relevances = []
    for doc_id in retrieved_docs[:k]:
        rel = qrels_for_query.get(doc_id, 0)
        relevances.append(rel)
    
    # Compute DCG
    dcg = compute_dcg(relevances)
    
    # Compute ideal DCG (IDCG) - sort all relevant documents by relevance
    all_relevances = sorted(qrels_for_query.values(), reverse=True)
    ideal_relevances = all_relevances[:k]
    idcg = compute_dcg(ideal_relevances)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def print_comparison_table(tfidf_ndcg_scores, bm25_ndcg_scores):
    print("nDCG@10 COMPARISON: TF-IDF vs BM25")
    print(f"{'Query ID':<12} {'TF-IDF':>10} {'BM25':>10} {'Diff':>10}")
    print("-"*60)
    
    tfidf_dict = dict(tfidf_ndcg_scores)
    bm25_dict = dict(bm25_ndcg_scores)
    
    for query_id in sorted(tfidf_dict.keys()):
        t_score = tfidf_dict[query_id]
        b_score = bm25_dict[query_id]
        diff = b_score - t_score
        print(f"{query_id:<12} {t_score:>10.4f} {b_score:>10.4f} {diff:>+10.4f}")
    
    print("-"*60)
    mean_tfidf = sum(tfidf_dict.values()) / len(tfidf_dict)
    mean_bm25 = sum(bm25_dict.values()) / len(bm25_dict)
    mean_diff = mean_bm25 - mean_tfidf
    print(f"{'MEAN':<12} {mean_tfidf:>10.4f} {mean_bm25:>10.4f} {mean_diff:>+10.4f}")

def print_divergent_query(query_id, query_text, tfidf_top10, bm25_top10, qrels_for_query, corpus, index, doc_term_freqs, doc_lengths, avgdl, N):
    print(f"DIVERGENT QUERY EXAMPLE: {query_id}")
    print(f"Query text: '{query_text}'")
    print(f"Query tokens: {tokenize(query_text)}")
    print(f"Corpus stats: N={N}, avgdl={avgdl:.2f}")
    print()
    
    tfidf_ids = [doc_id for doc_id, _ in tfidf_top10]
    bm25_ids = [doc_id for doc_id, _ in bm25_top10]
    
    print(f"{'Rank':<6} {'TF-IDF Doc':<15} {'Rel':>4} {'Score':>10} {'BM25 Doc':<15} {'Rel':>4} {'Score':>10} {'Same?':>6}")
    print("-"*100)
    
    tfidf_dict = dict(tfidf_top10)
    bm25_dict = dict(bm25_top10)
    
    for i in range(10):
        t_doc = tfidf_ids[i] if i < len(tfidf_ids) else "N/A"
        b_doc = bm25_ids[i] if i < len(bm25_ids) else "N/A"
        t_rel = qrels_for_query.get(t_doc, 0) if t_doc != "N/A" else 0
        b_rel = qrels_for_query.get(b_doc, 0) if b_doc != "N/A" else 0
        t_score = tfidf_dict.get(t_doc, 0) if t_doc != "N/A" else 0
        b_score = bm25_dict.get(b_doc, 0) if b_doc != "N/A" else 0
        same = "+" if t_doc == b_doc else "-"
        print(f"{i+1:<6} {t_doc:<15} {t_rel:>4} {t_score:>10.4f} {b_doc:<15} {b_rel:>4} {b_score:>10.4f} {same:>6}")
    
    # Show documents that appear in one ranking but not the other
    tfidf_set = set(tfidf_ids)
    bm25_set = set(bm25_ids)
    
    only_in_tfidf = tfidf_set - bm25_set
    only_in_bm25 = bm25_set - tfidf_set
    
    if only_in_tfidf:
        print(f"\nDocuments only in TF-IDF top-10: {', '.join(only_in_tfidf)}")
    if only_in_bm25:
        print(f"Documents only in BM25 top-10: {', '.join(only_in_bm25)}")
    
    query_tokens = tokenize(query_text)
    
    all_diff_docs = only_in_tfidf | only_in_bm25
    for doc_id in all_diff_docs:
        print(f"\n--- Document: {doc_id} ---")
        print(f"  Relevance: {qrels_for_query.get(doc_id, 0)}")
        print(f"  Document length: {doc_lengths.get(doc_id, 0)} tokens")
        print(f"  Length vs avgdl: {doc_lengths.get(doc_id, 0) / avgdl:.3f}x")
        
        print(f"\n  Term frequencies in document:")
        doc_terms = doc_term_freqs.get(doc_id, {})
        for term in query_tokens:
            tf = doc_terms.get(term, 0)
            df = compute_df(index, term)
            idf_tfidf = compute_idf(N, df)
            idf_bm25 = compute_bm25_idf(N, df)
            print(f"    '{term}': tf={tf}, df={df}, IDF_tfidf={idf_tfidf:.4f}, IDF_bm25={idf_bm25:.4f}")
        
        doc_len = doc_lengths.get(doc_id, 0)
        
        # Show why BM25 ranked it differently
        if doc_id in only_in_bm25:
            print(f"  BM25 PROMOTED this document (relevance={qrels_for_query.get(doc_id, 0)})")
            if doc_len < avgdl:
                print(f"      Reason: Document is shorter than average ({doc_len} < {avgdl:.0f}), boosted by length normalization")
        elif doc_id in only_in_tfidf:
            print(f"  TF-IDF kept this document but BM25 DEMOTED it (relevance={qrels_for_query.get(doc_id, 0)})")
            if doc_len > avgdl:
                print(f"      Reason: Document is longer than average ({doc_len} > {avgdl:.0f}), penalized by length normalization")
    

if __name__ == "__main__":
    
    print(f"\nDataset loaded (beir/nfcorpus/test)")
    print(f"    - Corpus size: {len(docs)} documents")
    print(f"    - Total queries: {len(queries)}")
    print(f"    - Queries with judgments: {len(qrels)}")
    
    # Build fixed query set: first 30 by query_id
    sorted_query_ids = sorted(queries.keys())
    query_set_ids = sorted_query_ids[:30]
    print(f"\nFixed set: first {len(query_set_ids)} queries sorted by query_id)")
    
    # Build inverted index
    print("\nBuilding inverted index")
    index, doc_term_freqs, doc_lengths, avgdl, N = build_inverted_index(docs)
    print(f"    - Vocabulary size: {len(index)} unique terms")
    print(f"    - Average document length: {avgdl:.2f} tokens")
    
    # Rank documents for each query
    print("\nRanking documents with TF-IDF and BM25")
    tfidf_results = {}  # query_id -> ranked list of (doc_id, score)
    bm25_results = {}   # query_id -> ranked list of (doc_id, score)
    
    for query_id in query_set_ids:
        query_text = queries[query_id]
        query_tokens = tokenize(query_text)
        
        # TF-IDF ranking
        tfidf_ranking = rank_documents_tfidf(query_tokens, docs, index, doc_term_freqs, N)
        tfidf_results[query_id] = tfidf_ranking
        
        # BM25 ranking
        bm25_ranking = rank_documents_bm25(query_tokens, docs, index, doc_term_freqs, doc_lengths, avgdl, N)
        bm25_results[query_id] = bm25_ranking
    
    # Compute nDCG@10
    print("\nComputing nDCG@10")
    tfidf_ndcg_scores = []
    bm25_ndcg_scores = []
    
    for query_id in query_set_ids:
        if query_id not in qrels:
            continue
        
        # Get top-10 for TF-IDF
        tfidf_top10 = [doc_id for doc_id, _ in tfidf_results[query_id][:10]]
        tfidf_ndcg = compute_ndcg_at_k(tfidf_top10, qrels[query_id], k=10)
        tfidf_ndcg_scores.append((query_id, tfidf_ndcg))
        
        # Get top-10 for BM25
        bm25_top10 = [doc_id for doc_id, _ in bm25_results[query_id][:10]]
        bm25_ndcg = compute_ndcg_at_k(bm25_top10, qrels[query_id], k=10)
        bm25_ndcg_scores.append((query_id, bm25_ndcg))
    
    # Print comparison table
    print_comparison_table(tfidf_ndcg_scores, bm25_ndcg_scores)
    
    # Find and display a query where rankings differ
    print("\nFinding a query where TF-IDF and BM25 rank differently")
    
    differing_query_id = None
    for query_id in query_set_ids:
        if query_id not in qrels:
            continue
            
        tfidf_top10 = [doc_id for doc_id, _ in tfidf_results[query_id][:10]]
        bm25_top10 = [doc_id for doc_id, _ in bm25_results[query_id][:10]]
        
        if tfidf_top10 != bm25_top10:
            differing_query_id = query_id
            break
    
    if differing_query_id:
        print_divergent_query(
            differing_query_id,
            queries[differing_query_id],
            tfidf_results[differing_query_id][:10],
            bm25_results[differing_query_id][:10],
            qrels[differing_query_id],
            docs,
            index,
            doc_term_freqs,
            doc_lengths,
            avgdl,
            N
        )