import math
import re  # for tokenization
import json
from collections import defaultdict

# Load data
def load_corpus(path):
    corpus = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc['_id']] = {
                'title': doc.get('title', ''),
                'text': doc.get('text', '')
            }
    return corpus

def load_queries(path):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line.strip())
            queries.append({
                '_id': q['_id'],
                'text': q.get('text', '')
            })
    return queries

def load_qrels(path):
    qrels = defaultdict(dict)
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                qrels[query_id][doc_id] = score
    return qrels

def tokenize(text):
    lowercased = text.lower()
    stripped = re.sub(r'[^\w\s]', '', lowercased)
    tokenized = stripped.split()
    return tokenized

def build_inverted_index(corpus):
    index = defaultdict(list)  # term -> [(doc_id, tf), ...]
    doc_term_freqs = {}  # doc_id -> {term: tf}
    doc_lengths = {}
    
    # compute tf for each document
    for doc_id, doc in corpus.items():
        full_text = doc['title'] + ' ' + doc['text']
        tokens = tokenize(full_text)
        
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
    
    return dict(index), doc_lengths, avgdl, N

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
        
        # TF-IDF weight
        weight = tf * idf
        score += weight
    return score

def rank_documents_tfidf(query_tokens, corpus, index, N):
    # Pre-compute tf for each document
    doc_term_freqs = {}
    for doc_id in corpus:
        full_text = corpus[doc_id]['title'] + ' ' + corpus[doc_id]['text']
        tokens = tokenize(full_text)
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        doc_term_freqs[doc_id] = term_freqs
    
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

def compute_bm25_score(query_tokens, doc_id, index, doc_lengths, avgdl, N, k1=1.5, b=0.75):
    score = 0.0
    doc_len = doc_lengths.get(doc_id, 0)
    
    # Get tf for this document from index
    doc_term_freqs = {}
    for term in query_tokens:
        if term in index:
            for d_id, tf in index[term]:
                if d_id == doc_id:
                    doc_term_freqs[term] = tf
                    break
    
    for term in query_tokens:
        f = doc_term_freqs.get(term, 0)
        if f == 0:
            continue
        
        # Compute BM25 IDF
        df = compute_df(index, term)
        idf = compute_bm25_idf(N, df)
        
        B = ((k1 + 1) * f) / (k1 * ((1 - b) + b * (doc_len / avgdl)) + f)
        score += idf * B
    return score

def rank_documents_bm25(query_tokens, corpus, index, doc_lengths, avgdl, N, k1=1.5, b=0.75):
    scores = []
    for doc_id in corpus:
        score = compute_bm25_score(query_tokens, doc_id, index, doc_lengths, avgdl, N, k1, b)
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
    # Compute ideal DCG (IDCG)
    # Sort all relevant documents by relevance
    all_relevances = sorted(qrels_for_query.values(), reverse=True)
    ideal_relevances = all_relevances[:k]
    idcg = compute_dcg(ideal_relevances)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

if __name__ == "__main__":
    # Load data
    corpus = load_corpus('lab_01/nfcorpus/corpus.jsonl')
    all_queries = load_queries('lab_01/nfcorpus/queries.jsonl')
    qrels = load_qrels('lab_01/nfcorpus/qrels/test.tsv')
    
    print(f"  - Corpus size: {len(corpus)} documents")
    print(f"  - Total queries: {len(all_queries)}")
    print(f"  - Qrels entries: {len(qrels)} queries with judgments")
    
    # Sort by query_id and take first 30
    sorted_queries = sorted(all_queries, key=lambda x: x['_id'])
    query_set = sorted_queries[:30]
    print(f"  - Fixed query set: {len(query_set)} queries (first 30 by query_id)")
    
    index, doc_lengths, avgdl, N = build_inverted_index(corpus)
    print(f"  - Vocabulary size: {len(index)} unique terms")
    print(f"  - Average document length: {avgdl:.2f} tokens")
    print(f"  - Total documents (N): {N}")
    
    tfidf_results = {}  # query_id -> ranked list of (doc_id, score)
    bm25_results = {}   # query_id -> ranked list of (doc_id, score)
    
    for query in query_set:
        query_id = query['_id']
        query_text = query['text']
        query_tokens = tokenize(query_text)
        
        # TF-IDF ranking
        tfidf_ranking = rank_documents_tfidf(query_tokens, corpus, index, N)
        tfidf_results[query_id] = tfidf_ranking
        
        # BM25 ranking
        bm25_ranking = rank_documents_bm25(query_tokens, corpus, index, doc_lengths, avgdl, N)
        bm25_results[query_id] = bm25_ranking
    
    print("\n[4] Computing nDCG@10...")
    
    tfidf_ndcg_scores = []
    bm25_ndcg_scores = []
    
    for query in query_set:
        query_id = query['_id']
        
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
    
    # Compute mean nDCG@10
    mean_tfidf_ndcg = sum(score for _, score in tfidf_ndcg_scores) / len(tfidf_ndcg_scores) if tfidf_ndcg_scores else 0
    mean_bm25_ndcg = sum(score for _, score in bm25_ndcg_scores) / len(bm25_ndcg_scores) if bm25_ndcg_scores else 0
    
    print(f"\n  RESULTS:")
    print(f"  - Mean nDCG@10 (TF-IDF): {mean_tfidf_ndcg:.4f}")
    print(f"  - Mean nDCG@10 (BM25):   {mean_bm25_ndcg:.4f}")
    
    # Find a query where they rank differently
    print("\n[5] Finding a query where TF-IDF and BM25 rank differently...")
    
    differing_query = None
    for query in query_set:
        query_id = query['_id']
        
        tfidf_top10 = [doc_id for doc_id, _ in tfidf_results[query_id][:10]]
        bm25_top10 = [doc_id for doc_id, _ in bm25_results[query_id][:10]]
        
        if tfidf_top10 != bm25_top10:
            differing_query = query
            break
    
    if differing_query:
        query_id = differing_query['_id']
        query_text = differing_query['text']
        
        print(f"\n  EXAMPLE QUERY: {query_id}")
        print(f"  Query text: '{query_text}'")
        
        tfidf_top10 = tfidf_results[query_id][:10]
        bm25_top10 = bm25_results[query_id][:10]
        
        print(f"\n  TF-IDF Top-10:")
        for i, (doc_id, score) in enumerate(tfidf_top10, 1):
            rel = qrels.get(query_id, {}).get(doc_id, 0)
            print(f"    {i}. {doc_id} (score={score:.4f}, relevance={rel})")
        
        print(f"\n  BM25 Top-10:")
        for i, (doc_id, score) in enumerate(bm25_top10, 1):
            rel = qrels.get(query_id, {}).get(doc_id, 0)
            print(f"    {i}. {doc_id} (score={score:.4f}, relevance={rel})")
        
    else:
        print("  No query with different rankings found")