import math
import re
import numpy as np
import ir_datasets
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder


def load_nfcorpus():
    ds = ir_datasets.load("beir/nfcorpus/test")

    corpus = {}
    for doc in ds.docs_iter():
        corpus[doc.doc_id] = {
            "title": "",
            "text": doc.text
        }

    queries = []
    for q in ds.queries_iter():
        queries.append({
            "_id": q.query_id,
            "text": q.text
        })

    qrels = defaultdict(dict)
    for qr in ds.qrels_iter():
        qrels[qr.query_id][qr.doc_id] = qr.relevance

    return corpus, queries, qrels

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def build_inverted_index(corpus):
    index = defaultdict(list)
    doc_lengths = {}
    doc_tf = {}

    for doc_id, doc in corpus.items():
        text = doc["title"] + " " + doc["text"]
        tokens = tokenize(text)
        doc_lengths[doc_id] = len(tokens)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        doc_tf[doc_id] = tf

    for doc_id, tf in doc_tf.items():
        for term, freq in tf.items():
            index[term].append((doc_id, freq))

    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    return index, doc_lengths, avgdl

def bm25_score(query_tokens, doc_id, index, doc_lengths, avgdl, N, k1=1.5, b=0.75):
    score = 0
    doc_len = doc_lengths[doc_id]

    for term in query_tokens:
        postings = index.get(term, [])
        tf = 0
        for d, freq in postings:
            if d == doc_id:
                tf = freq
                break
        if tf == 0:
            continue
        df = len(postings)
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        numerator = (k1 + 1) * tf
        denominator = k1 * ((1 - b) + b * doc_len / avgdl) + tf
        score += idf * numerator / denominator
    return score

def bm25_retrieve(query, index, corpus, doc_lengths, avgdl, top_k=100):
    tokens = tokenize(query)
    N = len(corpus)
    scores = []
    for doc_id in corpus:
        s = bm25_score(tokens, doc_id, index, doc_lengths, avgdl, N)
        scores.append((doc_id, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def dcg(rels):
    value = 0
    for i, rel in enumerate(rels, 1):
        gain = 2 ** rel - 1
        value += gain / math.log2(i + 1)
    return value

def ndcg10(ranking, qrels):
    rels = []
    for doc_id, _ in ranking[:10]:
        rels.append(qrels.get(doc_id, 0))

    ideal = sorted(qrels.values(), reverse=True)[:10]
    if dcg(ideal) == 0:
        return 0
    return dcg(rels) / dcg(ideal)

class DenseScout:
    def __init__(self, corpus):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.ids = list(corpus.keys())
        texts = [corpus[i]["text"] for i in self.ids]
        emb = self.model.encode(texts, show_progress_bar=True)
        self.emb = self.normalize(emb)

    def normalize(self, x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def retrieve(self, query, top_k=100):
        q = self.model.encode([query])
        q = self.normalize(q)
        scores = (self.emb @ q.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in idx]

class Judge:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, candidates, corpus, top_k=10):
        pairs = []
        for doc_id, _ in candidates:
            pairs.append((query, corpus[doc_id]["text"]))

        scores = self.model.predict(pairs)
        result = [(candidates[i][0], float(scores[i])) for i in range(len(scores))]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]

if __name__ == "__main__":
    corpus, queries, qrels = load_nfcorpus()

    print("=" * 60)

    index, doc_lengths, avgdl = build_inverted_index(corpus)

    query_set = sorted(queries, key=lambda x: x["_id"])[:30]

    dense = DenseScout(corpus)
    judge = Judge()

    print("\nPre-computing retrievals (top-100) for 30 queries")
    precomputed = []
    for q in query_set:
        qid = q["_id"]
        text = q["text"]
        bm25_100 = bm25_retrieve(text, index, corpus, doc_lengths, avgdl, 100)
        dense_100 = dense.retrieve(text, 100)
        precomputed.append({
            "qid": qid,
            "text": text,
            "bm25": bm25_100,
            "dense": dense_100
        })

    ladder = []
    print("\nEvaluation\n")

    for item in precomputed:
        qid = item["qid"]
        bm25 = item["bm25"]
        dense_rank = item["dense"]

        bm25_ndcg = ndcg10(bm25, qrels[qid])
        dense_ndcg = ndcg10(dense_rank, qrels[qid])

        bm25_ce = judge.rerank(item["text"], bm25[:50], corpus)
        dense_ce = judge.rerank(item["text"], dense_rank[:50], corpus)

        ladder.append([
            qid,
            bm25_ndcg,
            dense_ndcg,
            ndcg10(bm25_ce, qrels[qid]),
            ndcg10(dense_ce, qrels[qid])
        ])

    print("\nLIFT LADDER")
    print("Query\tBM25\tDense\tBM25+CE\tDense+CE")
    for r in ladder:
        print(r[0], *[f"{x:.4f}" for x in r[1:]], sep="\t")

    print("\nMEAN")
    mean_bm25 = np.mean([x[1] for x in ladder])
    mean_dense = np.mean([x[2] for x in ladder])
    mean_bm25_ce = np.mean([x[3] for x in ladder])
    mean_dense_ce = np.mean([x[4] for x in ladder])
    print(f"BM25:     {mean_bm25:.4f}")
    print(f"Dense:    {mean_dense:.4f}")
    print(f"BM25+CE:  {mean_bm25_ce:.4f}")
    print(f"Dense+CE: {mean_dense_ce:.4f}")

    # Which scout feeds the Judge better?
    print("\n" + "=" * 60)
    if mean_bm25_ce > mean_dense_ce:
        better = "BM25"
        print("BM25 feeds the Judge a better shortlist.")
    else:
        better = "Dense"
        print("Dense Scout feeds the Judge a better shortlist.")
    print(f"BM25+CE nDCG@10 = {mean_bm25_ce:.4f} vs Dense+CE = {mean_dense_ce:.4f}")
    print("=" * 60)

    # Find a query where rerank pulled up a buried doc
    print("\n--- Example: Rerank pulling up a buried relevant document ---\n")

    example_found = False
    example_query = None
    example_doc = None
    example_before = None
    example_after = None
    example_scout = None

    for item in precomputed:
        if example_found:
            break
        qid = item["qid"]
        text = item["text"]
        bm25 = item["bm25"]
        bm25_ce = judge.rerank(text, bm25[:50], corpus)

        bm25_top10_ids = {doc_id for doc_id, _ in bm25[:10]}
        ce_top10_ids = [doc_id for doc_id, _ in bm25_ce[:10]]

        for doc_id, _ in bm25[:50]:
            rel = qrels[qid].get(doc_id, 0)
            if rel > 0 and doc_id not in bm25_top10_ids and doc_id in ce_top10_ids:
                before_rank = next(i + 1 for i, (d, _) in enumerate(bm25) if d == doc_id)
                after_rank = next(i + 1 for i, d in enumerate(ce_top10_ids) if d == doc_id)

                print(f"Query ID: {qid}")
                print(f"Query:    {text}")
                print(f"Doc ID:   {doc_id}")
                print(f"Relevance: {rel}")
                print(f"BM25 rank:  {before_rank} (outside top-10)")
                print(f"After CE rerank: rank {after_rank} (inside top-10)")
                print()

                example_found = True
                example_query = qid
                example_doc = doc_id
                example_before = before_rank
                example_after = after_rank
                example_scout = "BM25"
                break

    if not example_found:
        print("No example found where a relevant doc was pulled from outside BM25 top-10 into CE top-10.")
        print("Checking dense scout instead...")
        for item in precomputed:
            if example_found:
                break
            qid = item["qid"]
            text = item["text"]
            dense_rank = item["dense"]
            dense_ce = judge.rerank(text, dense_rank[:50], corpus)

            dense_top10_ids = {doc_id for doc_id, _ in dense_rank[:10]}
            ce_top10_ids = [doc_id for doc_id, _ in dense_ce[:10]]

            for doc_id, _ in dense_rank[:50]:
                rel = qrels[qid].get(doc_id, 0)
                if rel > 0 and doc_id not in dense_top10_ids and doc_id in ce_top10_ids:
                    before_rank = next(i + 1 for i, (d, _) in enumerate(dense_rank) if d == doc_id)
                    after_rank = next(i + 1 for i, d in enumerate(ce_top10_ids) if d == doc_id)

                    print(f"Query ID: {qid}")
                    print(f"Query:    {text}")
                    print(f"Doc ID:   {doc_id}")
                    print(f"Relevance: {rel}")
                    print(f"Dense rank:  {before_rank} (outside top-10)")
                    print(f"After CE rerank: rank {after_rank} (inside top-10)")
                    print()

                    example_found = True
                    example_query = qid
                    example_doc = doc_id
                    example_before = before_rank
                    example_after = after_rank
                    example_scout = "Dense"
                    break

    # Rerank-depth sweep (reuses precomputed top-100)
    depths = [5, 10, 20, 50, 100]

    bm25_sweep = []
    dense_sweep = []

    print("\nRerank-depth sweep (reusing precomputed top-100)")
    for depth in depths:
        bm25_scores = []
        dense_scores = []

        for item in precomputed:
            qid = item["qid"]
            text = item["text"]

            bm25_reranked = judge.rerank(text, item["bm25"][:depth], corpus)
            bm25_scores.append(ndcg10(bm25_reranked, qrels[qid]))

            dense_reranked = judge.rerank(text, item["dense"][:depth], corpus)
            dense_scores.append(ndcg10(dense_reranked, qrels[qid]))

        bm25_sweep.append(np.mean(bm25_scores))
        dense_sweep.append(np.mean(dense_scores))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(depths, bm25_sweep, marker="o", label="BM25 → CE")
    ax1.plot(depths, dense_sweep, marker="s", label="Dense → CE")
    ax1.set_xlabel("Rerank depth k")
    ax1.set_ylabel("nDCG@10")
    ax1.set_title("Cross-encoder depth sweep")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("CE forward passes per query (= k)")
    ax2.set_ylim(0, 120)
    ax2.tick_params(axis="y")

    plt.tight_layout()
    plt.savefig("rerank_depth_sweep.png")
    print("Plot saved to rerank_depth_sweep.png")

    print("\n--- Depth sweep results ---")
    print("k\tBM25→CE\tDense→CE")
    for i, k in enumerate(depths):
        print(f"{k}\t{bm25_sweep[i]:.4f}\t{dense_sweep[i]:.4f}")

    print("\nObservation:")
    saturation_k = None
    for i in range(1, len(depths)):
        bm25_gain = bm25_sweep[i] - bm25_sweep[i - 1]
        dense_gain = dense_sweep[i] - dense_sweep[i - 1]
        if bm25_gain < 0.001 and dense_gain < 0.001:
            saturation_k = depths[i - 1]
            print(f"Going deeper than k={depths[i-1]} stops paying off — gains become negligible.")
            break
    else:
        print("Gains are still visible at k=100; deeper might help but cost rises linearly.")