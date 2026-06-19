Put a fast dense Scout and a careful cross-encoder Judge on top of the BM25 you built in Lab 1, wire them into a retrieve→rerank cascade, and measure — with your own numbers — exactly how much the rerank buys and what it costs.

Tasks
Load BEIR nfcorpus/test and the fixed query set (first 30 by query_id). Reuse your Lab-1 BM25 and nDCG@10 implementations as the first stage and the metric.
Dense Scout (bi-encoder): encode queries and documents with a pretrained SBERT (sentence-transformers/all-MiniLM-L6-v2), L2-normalize, and retrieve the top-100 by cosine. The embeddings come from the model; the cosine search is yours.
Cross-encoder Judge: score (query, document) pairs jointly with a pretrained cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) — read each pair together, nothing precomputed.
The cascade: for each query take the first-stage top-k and rerank it with the Judge, emitting a reranked top-10. Build it for both first stages — BM25→rerank and dense→rerank.
The lift ladder: report nDCG@10 at three stages — BM25 alone, dense alone, and +cross-encoder rerank — in one table. Which Scout feeds the Judge a better shortlist?
Rerank-depth sweep: vary the rerank depth k ∈ {5, 10, 20, 50, 100}; plot nDCG@10 vs k and the cost (cross-encoder passes per query = k). Where does going deeper stop paying off?
Show one query where the rerank pulled a buried relevant document up into the top-10 — give the before and after ranks.
Deliverables
lab.py (or a notebook) that runs end-to-end and prints the lift-ladder table