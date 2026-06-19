Goal
Implement an inverted index, TF-IDF, and BM25 from scratch on a real judged corpus, then compare TF-IDF against BM25 on a fixed query set and explain — with a concrete example from your own output — where and why they differ.

Tasks
Load BEIR nfcorpus/test and build the fixed query set (first 30 by query_id).
Tokenize with one fixed scheme: lowercase → strip punctuation → split on whitespace. (The Lab holds tokenization fixed; the Homework varies it.)
Build an inverted index from scratch: term → postings [(doc_id, tf)]; also record document lengths and avgdl.
Implement TF-IDF by hand: w(t,d) = tf · log(N/df). State your tf and idf variant; score by summing query-term weights or by cosine (your choice, stated).
Implement BM25 by hand: k₁ = 1.5, b = 0.75, smoothed idf ln(1 + (N−df+0.5)/(df+0.5)). Reuse the same index.
For each query, retrieve the top-10 under TF-IDF and under BM25.
Compute nDCG@10 (by hand, graded gain 2^rel−1, log₂(i+1) discount) per query for both rankers; report the mean for each and show one query where they rank differently.
Write a half-page analysis: where does BM25 diverge from TF-IDF, and why? Point to term-frequency saturation and document-length normalization using your concrete example.

Deliverables
lab.py (or a notebook) that runs end-to-end and prints the comparison table, plus a 1-page report: the mean-nDCG@10 table, one divergent-query table, and the analysis paragraph.

## Assignment requirements
Use BEIR nfcorpus/test over the full test query set (optionally repeat the headline experiment on scifact/test).

Rankers (by hand): a Boolean-OR baseline, TF-IDF (cosine), BM25, and RRF fusion (Σ 1/(k+rank), k=60) of BM25 + TF-IDF.
Metrics (by hand): nDCG@10, MAP, MRR over all queries — report the mean and the per-query distribution (box/violin or histogram). Use graded gain 2^rel−1 and the log₂(i+1) discount.
BM25 parameter study: sweep k₁ ∈ {0.5,1.0,1.2,1.5,2.0,3.0} and b ∈ {0,0.25,0.5,0.75,1.0}; plot nDCG@10 vs k₁ (b fixed) and vs b (k₁ fixed); identify the best (k₁,b) and explain the shape (saturation; the b=0/1 extremes).
Tokenization study: compare ≥3 pipelines on the same BM25 — (a) lowercase+whitespace, (b) + stopword removal, (c) + Porter stemming, (optional d) subword/BPE. Report nDCG@10 / MAP / MRR for each and explain why each choice helps or hurts (vocabulary size, df shifts, conflation vs over-conflation). Tie back to Lecture 02 and 03.
Statistical significance: is your best system better than the BM25 baseline, or is it chance? Run a paired test (t-test and/or Wilcoxon signed-rank) on per-query nDCG@10; report the mean difference, a 95% CI, and the p-value, and interpret it (necessary-not-sufficient; the small-n caveat).
Report: plots + tables + a discussion connecting the parameter sweep, the tokenization study, and the significance test into one story about what actually moves ranking quality.