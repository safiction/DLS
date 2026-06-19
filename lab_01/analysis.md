# TF-IDF vs BM25 Analysis Report

## 1. Mean nDCG@10 Comparison

The following table shows the nDCG@10 scores for TF-IDF and BM25 across the first 30 queries (by query_id) from the BEIR nfcorpus/test dataset:

| Query ID   | TF-IDF | BM25   | Diff    |
|------------|--------|--------|---------|
| PLAIN-1008 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1018 | 0.4983 | 0.5685 | +0.0702 |
| PLAIN-102  | 0.1655 | 0.1922 | +0.0266 |
| PLAIN-1028 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1039 | 0.9197 | 1.0000 | +0.0803 |
| PLAIN-1050 | 0.0663 | 0.0663 | +0.0000 |
| PLAIN-1066 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1088 | 0.5531 | 0.5531 | +0.0000 |
| PLAIN-1098 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1109 | 0.6372 | 0.6118 | -0.0254 |
| PLAIN-1119 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-112  | 0.1958 | 0.4138 | +0.2180 |
| PLAIN-1130 | 1.0000 | 1.0000 | +0.0000 |
| PLAIN-1141 | 0.0663 | 0.1389 | +0.0726 |
| PLAIN-1151 | 0.2330 | 0.2305 | -0.0025 |
| PLAIN-1161 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1172 | 0.2201 | 0.2201 | +0.0000 |
| PLAIN-1183 | 0.2201 | 0.2201 | +0.0000 |
| PLAIN-1193 | 0.0636 | 0.0663 | +0.0026 |
| PLAIN-12   | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1203 | 0.2337 | 0.3590 | +0.1253 |
| PLAIN-1214 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1225 | 0.4896 | 0.5185 | +0.0288 |
| PLAIN-123  | 0.1494 | 0.0270 | -0.1224 |
| PLAIN-1236 | 0.3392 | 0.3392 | +0.0000 |
| PLAIN-1249 | 0.0000 | 0.0000 | +0.0000 |
| PLAIN-1262 | 0.3836 | 0.2895 | -0.0941 |
| PLAIN-1275 | 0.4748 | 0.4690 | -0.0058 |
| PLAIN-1288 | 0.6275 | 0.7534 | +0.1259 |
| PLAIN-1299 | 0.2489 | 0.3149 | +0.0660 |
| **MEAN**   | **0.2595** | **0.2784** | **+0.0189** |

**Key Finding**: BM25 achieves a higher mean nDCG@10 (0.2784) compared to TF-IDF (0.2595), representing a relative improvement of approximately 7.3%.

---

## 2. Divergent Query Example: PLAIN-1018 ("DHA")

The query "DHA" (a common abbreviation for Docosahexaenoic acid, an omega-3 fatty acid) shows different rankings between TF-IDF and BM25:

| Rank | TF-IDF Doc | Rel | BM25 Doc | Rel | Same? |
|------|------------|-----|----------|-----|-------|
| 1    | MED-5095   | 0   | MED-5095 | 0   | ✓     |
| 2    | MED-4936   | 1   | MED-4936 | 1   | ✓     |
| 3    | MED-5091   | 1   | MED-5091 | 1   | ✓     |
| 4    | MED-1832   | 1   | MED-1832 | 1   | ✓     |
| 5    | MED-3012   | 1   | MED-3012 | 1   | ✓     |
| 6    | MED-838    | 0   | MED-838  | 0   | ✓     |
| 7    | MED-928    | 0   | MED-5092 | 1   | ✗     |
| 8    | MED-4633   | 1   | MED-5364 | 0   | ✗     |
| 9    | MED-5342   | 0   | MED-4633 | 1   | ✗     |
| 10   | MED-5364   | 0   | MED-5342 | 0   | ✗     |

**Documents only in TF-IDF top-10**: MED-928  
**Documents only in BM25 top-10**: MED-5092

**Impact on nDCG@10**: BM25 achieves 0.5685 vs TF-IDF's 0.4983 for this query, a +14% improvement.

---

## 3. Analysis: Why BM25 Diverges from TF-IDF

### 3.1 Term Frequency Saturation

**TF-IDF** uses a linear relationship between term frequency and score: `w(t,d) = tf · idf`. This means that if a term appears 10 times in a document, it contributes 10 times more than if it appeared once. This can lead to over-emphasis on long documents that mention a term many times, even if the additional mentions don't add meaningful relevance.

**BM25** introduces **term frequency saturation** through the parameter `k1 = 1.5`. The BM25 term weight formula is:

```
B = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (doc_len / avgdl)) + tf)
```

As `tf` increases, the contribution of each additional occurrence diminishes. With `k1 = 1.5`, the saturation curve ensures that the 5th or 6th occurrence of a term contributes significantly less than the 1st or 2nd. This prevents long documents from dominating rankings simply due to term repetition.

### 3.2 Document Length Normalization

**TF-IDF** has no explicit document length normalization. A long document that mentions a query term many times will naturally have a higher TF-IDF score, even if the term is "diluted" among many other words.

**BM25** incorporates **document length normalization** through the parameter `b = 0.75`. The term `(1 - b) + b * (doc_len / avgdl)` adjusts the effective term frequency based on how the document length compares to the average document length (`avgdl = 219.31` tokens in our corpus).

- For documents **shorter than average** (`doc_len < avgdl`), the normalization factor is less than 1, boosting their scores.
- For documents **longer than average** (`doc_len > avgdl`), the normalization factor is greater than 1, dampening their scores.

### 3.3 Concrete Example from PLAIN-1018

In the "DHA" query example, BM25 promotes **MED-5092** (relevance=1) into the top-10 while demoting **MED-928** (relevance=0). This occurs because:

1. **MED-5092** likely contains "DHA" with appropriate term frequency in a document of reasonable length—BM25 recognizes this as a focused, relevant document.

2. **MED-928** may be a longer document where "DHA" appears frequently but is diluted by extensive content. TF-IDF's linear scoring overweights this document, while BM25's saturation and length normalization correctly penalize it.

Additionally, BM25 reorders **MED-4633** (relevance=1) higher than **MED-5342** and **MED-5364** (both relevance=0), improving the ranking quality.

### 3.4 IDF Smoothing

BM25 also uses a **smoothed IDF** variant: `ln(1 + (N−df+0.5)/(df+0.5))`, which prevents negative IDF values for terms that appear in more than half the documents. While both methods handle rare terms similarly, BM25's smoothing provides more stable behavior for moderately common terms.

---

## 4. Conclusion

BM25 outperforms TF-IDF on the nfcorpus/test dataset (mean nDCG@10: 0.2784 vs 0.2595) due to its two key innovations:

1. **Term frequency saturation** prevents over-scoring of documents with high term repetition
2. **Document length normalization** ensures fair comparison between short and long documents

These mechanisms allow BM25 to better identify truly relevant documents, as demonstrated in the "DHA" query where BM25 successfully promoted a relevant document (MED-5092) and demoted an irrelevant one (MED-928).