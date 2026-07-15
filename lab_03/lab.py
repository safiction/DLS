import json
import math
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset # EnterpriseRAG-Bench (onyx-dot-app) -- MIT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

SEED = 20260605
SLICE = "confluence" # approx 5.2K docs
SLICE_CAP = 5000
TOP_K = 10
RERANK_DEPTH = 100
DIM = 384
STORE_DIR = os.path.dirname(os.path.abspath(__file__))

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def banner(msg):
    print("\n" + "=" * 72, flush=True)
    print(msg)
    print("=" * 72, flush=True)

def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def first_30(qs):
    return sorted(qs, key=lambda q: q["question_id"])[:30]

def build_corpus(questions):
    keep = {i for q in questions for i in q["expected_doc_ids"]}
    log(f"{len(questions)} questions -> {len(keep)} unique gold doc_ids")

    cache = os.path.join(STORE_DIR, f"corpus_{SLICE.lower()}.jsonl")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]
        log(f"corpus cache hit: {cache} ({len(corpus)} docs) -- skipping the stream")
        return corpus, keep

    log("corpus cache miss -> streaming the documents config (~500K docs)")
    stream = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents",
                          split="test", streaming=True)

    gold, slice_docs, type_counts = [], [], {}
    t0 = time.time()
    for n, d in enumerate(stream, 1):
        st = d["source_type"]
        type_counts[st] = type_counts.get(st, 0) + 1
        if d["doc_id"] in keep:
            gold.append(d)
        elif st.lower() == SLICE.lower():
            slice_docs.append(d)
        if n % 25000 == 0:
            log(f"  streamed {n:>7,} docs | gold {len(gold)}/{len(keep)}"
                f" | {SLICE} slice {len(slice_docs):,} | {time.time()-t0:,.0f}s")
    log(f"stream done: {n:,} docs in {time.time()-t0:,.0f}s")
    log("source_type counts seen: " +
        ", ".join(f"{k}={v:,}" for k, v in sorted(type_counts.items())))

    missing = keep - {d["doc_id"] for d in gold}
    if missing:
        log(f"WARNING: {len(missing)} gold doc_ids never appeared in the stream: "
            f"{sorted(missing)[:5]}...")

    if len(slice_docs) > SLICE_CAP:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(slice_docs), SLICE_CAP, replace=False))
        log(f"sampling slice {len(slice_docs):,} -> {SLICE_CAP:,} (seed {SEED})")
        slice_docs = [slice_docs[i] for i in idx]

    corpus = gold + slice_docs
    fields = ("doc_id", "source_type", "title", "content")
    with open(cache, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps({k: d.get(k) for k in fields}) + "\n")
    log(f"corpus = {len(gold)} gold + {len(slice_docs):,} {SLICE} "
        f"= {len(corpus):,} docs -> cached to {cache}")
    return corpus, keep

def embed_all(corpus, questions):
    doc_npy = os.path.join(STORE_DIR, f"emb_docs_{SLICE.lower()}.npy")
    q_npy = os.path.join(STORE_DIR, f"emb_questions_{SLICE.lower()}.npy")

    if os.path.exists(doc_npy) and os.path.exists(q_npy):
        X, Q = np.load(doc_npy), np.load(q_npy)
        if len(X) == len(corpus) and len(Q) == len(questions):
            log(f"embedding cache hit: X {X.shape}, Q {Q.shape}")
            return X, Q
        log("embedding cache is stale (size mismatch) -> re-embedding")

    log("loading all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [((d.get("title") or "") + " " + (d.get("content") or "")).strip()
             for d in corpus]
    log(f"embedding {len(texts):,} documents (title+content)...")
    X = model.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True)
    log(f"embedding {len(questions)} questions...")
    Q = model.encode([q["question"] for q in questions], batch_size=64,
                     convert_to_numpy=True)

    X, Q = normalize(X.astype(np.float32)), normalize(Q.astype(np.float32))
    np.save(doc_npy, X)
    np.save(q_npy, Q)
    log(f"unit vectors cached: X {X.shape} -> {doc_npy}")
    return X, Q

def exact_baseline(X, Q, questions, doc_ids):
    scores = Q @ X.T
    exact_top10 = np.argsort(-scores, axis=1)[:, :TOP_K]

    id_pos = {d: i for i, d in enumerate(doc_ids)}
    hits, total = 0, 0
    for qi, q in enumerate(questions):
        gold_idx = {id_pos[d] for d in q["expected_doc_ids"] if d in id_pos}
        total += len(gold_idx)
        hits += len(gold_idx & set(exact_top10[qi]))
    log(f"sanity check vs expected_doc_ids: {hits}/{total} gold docs "
        f"inside the exact top-10 ({hits/max(total,1):.2f}) -- "
        "retrieval quality, unrelated to ANN recall below")
    return exact_top10


def recall_at_10(approx_top10, exact_top10):
    per_q = [len(set(a) & set(e)) / TOP_K
             for a, e in zip(approx_top10, exact_top10)]
    return float(np.mean(per_q)), per_q

class IVFIndex:
    """Partitioning index: k-means Voronoi cells + inverted lists.
    Query = rank against nlist centroids, exact-scan only top-nprobe lists"""

    def __init__(self, X, nlist):
        self.X, self.nlist = X, nlist
        log(f"IVF: fitting KMeans nlist={nlist} (~sqrt(N)), seed {SEED}...")
        km = KMeans(n_clusters=nlist, random_state=SEED, n_init=10)
        self.assign = km.fit_predict(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.lists = {c: np.where(self.assign == c)[0] for c in range(nlist)}
        sizes = [len(v) for v in self.lists.values()]
        log(f"IVF: inverted lists built, sizes min/median/max = "
            f"{min(sizes)}/{int(np.median(sizes))}/{max(sizes)}")

    def cell_order(self, q):
        d2 = ((self.centroids - q) ** 2).sum(axis=1)   # kmeans geometry
        return np.argsort(d2)

    def search(self, q, nprobe):
        cells = self.cell_order(q)[:nprobe]
        cand = np.concatenate([self.lists[c] for c in cells])
        s = self.X[cand] @ q                           # exact cosine on survivors
        top = cand[np.argsort(-s)[:TOP_K]]
        return top, len(cand)


def ivf_sweep(ivf, Q, exact_top10, N):
    nprobes = [1, 2, 4, 8, 16, ivf.nlist]
    rows = []
    for nprobe in nprobes:
        tops, scanned = [], []
        for qi in range(len(Q)):
            top, n_cand = ivf.search(Q[qi], nprobe)
            tops.append(top)
            scanned.append(n_cand / N)
        rec, _ = recall_at_10(tops, exact_top10)
        rows.append({"nprobe": nprobe, "recall": rec,
                     "frac_scanned": float(np.mean(scanned))})
        log(f"  nprobe={nprobe:>3}  recall@10={rec:.3f}  "
            f"fraction scanned={np.mean(scanned):.3f}")

    max_rec = rows[-1]["recall"] # nprobe=nlist == exact
    knee = next(r["nprobe"] for r in rows if r["recall"] >= 0.90 * max_rec)
    log(f"knee: recall saturates at nprobe={knee} "
        f"({0.90:.2f} x max recall {max_rec:.3f})")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = [r["nprobe"] for r in rows]
    ax1.plot(xs, [r["recall"] for r in rows], marker="o", label="recall@10")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("nprobe")
    ax1.set_ylabel("recall@10 vs exact top-10")
    ax1.axvline(knee, color="gray", ls="--", label=f"knee (nprobe={knee})")
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["frac_scanned"] for r in rows], marker="s", color="tab:red",
             label="fraction scanned")
    ax2.set_ylabel("fraction of corpus scanned")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("IVF: recall vs work (nprobe sweep)")
    plt.tight_layout()
    out = os.path.join(STORE_DIR, "ivf_nprobe_sweep.png")
    plt.savefig(out)
    plt.close()
    log(f"plot saved in -> {out}")
    return rows, knee


def boundary_example(ivf, Q, questions, exact_top10, doc_ids, corpus):
    best = None
    for qi in range(len(Q)):
        order = ivf.cell_order(Q[qi])
        top1, _ = ivf.search(Q[qi], 1)
        missed = set(exact_top10[qi]) - set(top1)
        for doc in missed:
            cell_rank = int(np.where(order == ivf.assign[doc])[0][0]) + 1
            exact_rank = int(np.where(exact_top10[qi] == doc)[0][0]) + 1
            cand = (qi, doc, cell_rank, exact_rank)
            if best is None or cell_rank < best[2]:
                best = cand
        if best is not None and best[2] == 2:
            break

    if best is None:
        log("no nprobe=1 miss found (recall already 1.0) -- nothing to exhibit")
        return None

    qi, doc, cell_rank, exact_rank = best
    d2 = ((ivf.centroids - Q[qi]) ** 2).sum(axis=1)
    order = ivf.cell_order(Q[qi])
    margin = float(d2[ivf.assign[doc]] - d2[order[0]])
    title = (corpus[doc].get("title") or "")[:80]
    print(f"""
--- boundary-neighbour failure (IVF, nprobe=1) ---
question ({questions[qi]['question_id']}): {questions[qi]['question'][:100]}
missed doc:   {doc_ids[doc]}  "{title}"
exact rank:   {exact_rank} (a true top-10 neighbour)
its cell is the query's centroid #{cell_rank} (nprobe=1 only scans #1);
centroid-distance margin to the winning cell: {margin:.4f}
-> the neighbour sits just across the Voronoi boundary; nprobe>={cell_rank} recovers it
""", flush=True)
    return {"question_id": questions[qi]["question_id"], "doc_id": doc_ids[doc],
            "exact_rank": exact_rank, "cell_rank": cell_rank, "margin": margin}


def lloyd_kmeans(data, k, seed, n_iter=30, verbose=False):
    rng = np.random.default_rng(seed)
    centroids = data[rng.choice(len(data), size=k, replace=False)].copy()
    prev = None
    for it in range(n_iter):
        d2 = ((data ** 2).sum(1)[:, None] + (centroids ** 2).sum(1)[None, :]
              - 2.0 * data @ centroids.T)
        assign = d2.argmin(1)
        inertia = float(d2[np.arange(len(data)), assign].sum())
        if verbose:
            log(f"    Lloyd iter {it:2d}: inertia = {inertia:,.2f}")
        for c in range(k):
            mask = assign == c
            if mask.any():
                centroids[c] = data[mask].mean(0)
            else:                                      # empty cluster -> re-seed
                centroids[c] = data[rng.integers(len(data))]
        if prev is not None and prev - inertia < 1e-4 * abs(prev):
            break
        prev = inertia
    return centroids, inertia


def pack_codes(codes, k):
    bits = int(math.log2(k))
    c = codes.astype(np.uint8)
    if bits == 8:
        return c
    if bits == 4:
        if c.shape[1] % 2:
            c = np.hstack([c, np.zeros((len(c), 1), np.uint8)])
        return (c[:, 0::2] << 4) | c[:, 1::2]
    raise ValueError(f"unsupported k={k}")


class PQIndex:
    """Compression index: m contiguous subvectors, one k-centroid codebook
    per subspace, ADC scoring (m table lookups + m-1 adds, no decompression)."""
    def __init__(self, X, m, k, verbose=False):
        self.m, self.k = m, k
        self.dsub = X.shape[1] // m
        assert X.shape[1] % m == 0
        log(f"PQ(m={m}, k={k}): fitting {m} codebooks "
            f"({self.dsub}-dim subspaces), seed {SEED}...")
        self.codebooks = np.empty((m, k, self.dsub), dtype=np.float32)
        self.codes = np.empty((len(X), m), dtype=np.uint8)
        for s in range(m):
            sub = X[:, s * self.dsub:(s + 1) * self.dsub]
            if verbose and s == 0:
                log("  falling inertia, subspace 0:")
            cb, inertia = lloyd_kmeans(sub, k, seed=SEED, verbose=(verbose and s == 0))
            self.codebooks[s] = cb
            d2 = ((sub ** 2).sum(1)[:, None] + (cb ** 2).sum(1)[None, :]
                  - 2.0 * sub @ cb.T)
            self.codes[:, s] = d2.argmin(1).astype(np.uint8)
            if verbose:
                log(f"  subspace {s:2d}/{m}: final inertia {inertia:,.2f}")

    def adc_scores(self, q):
        lut = np.einsum("mkd,md->mk",
                        self.codebooks,
                        q.reshape(self.m, self.dsub))          # (m, k) table
        return lut[np.arange(self.m), self.codes].sum(axis=1)  # (N,)

    def packed_bytes_per_vec(self):
        return self.m * math.log2(self.k) / 8


def pq_eval(pq, X, Q, exact_top10):
    """recall@10 with and without the exact re-rank of the PQ top-100."""
    raw_tops, rr_tops = [], []
    for qi in range(len(Q)):
        s = pq.adc_scores(Q[qi])
        order = np.argsort(-s)
        raw_tops.append(order[:TOP_K])                 # PQ alone
        short = order[:RERANK_DEPTH]                   # cheap retrieve
        s_exact = X[short] @ Q[qi]                     # exact re-rank
        rr_tops.append(short[np.argsort(-s_exact)[:TOP_K]])
    raw, _ = recall_at_10(raw_tops, exact_top10)
    rr, _ = recall_at_10(rr_tops, exact_top10)
    return raw, rr


def pq_memory_report(pq, N):
    packed = pack_codes(pq.codes, pq.k)
    actual = packed.nbytes / N
    formula = pq.packed_bytes_per_vec()
    log(f"  bytes/vector: uint8 array = {pq.codes.nbytes / N:.1f}, "
        f"PACKED = {actual:.1f}, formula m*log2(k)/8 = {formula:.1f}  "
        f"(compression {DIM * 4 / formula:.0f}x vs {DIM * 4} float32 bytes)")
    return formula

def main():
    banner("stage 1/8 -- questions + ~5K-doc corpus (stream, cached)")
    qs = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")["test"]
    questions = first_30(qs)
    corpus, _ = build_corpus(questions)
    doc_ids = [d["doc_id"] for d in corpus]
    N = len(corpus)
    results = {"seed": SEED, "slice": SLICE, "n_docs": N,
               "n_questions": len(questions)}

    banner("stage 2/8 -- SBERT embeddings (cached)")
    X, Q = embed_all(corpus, questions)

    banner("stage 3/8 -- exact baseline: flat cosine top-10 (ground truth)")
    exact_top10 = exact_baseline(X, Q, questions, doc_ids)
    log("exact top-10 recorded for every question; "
        "all recall@10 below is measured against it")

    banner("stage 4/8 -- IVF by hand + nprobe sweep + boundary failure")
    nlist = round(math.sqrt(N))
    ivf = IVFIndex(X, nlist)
    ivf_rows, knee = ivf_sweep(ivf, Q, exact_top10, N)
    results["ivf"] = {"nlist": nlist, "sweep": ivf_rows, "knee_nprobe": knee}
    results["boundary_example"] = boundary_example(
        ivf, Q, questions, exact_top10, doc_ids, corpus)

    banner("stage 5/8 -- PQ by hand (m=8, k=256): codebooks, ADC, re-rank")
    pq = PQIndex(X, m=8, k=256, verbose=True)
    pq_bytes = pq_memory_report(pq, N)
    raw, rr = pq_eval(pq, X, Q, exact_top10)
    log(f"PQ(8,256) recall@10: {raw:.3f} ALONE (hard ceiling -- loss is baked "
        f"into the codes) vs {rr:.3f} after exact re-rank of top-{RERANK_DEPTH}")
    results["pq"] = {"m": 8, "k": 256, "bytes_per_vec": pq_bytes,
                     "recall_raw": raw, "recall_rerank": rr}

    banner("stage 6/8 -- (m, k) memory sweep")
    sweep = []
    for m in (4, 8, 16):
        for k in (16, 256):
            p = pq if (m, k) == (8, 256) else PQIndex(X, m=m, k=k)
            b = pq_memory_report(p, N)
            raw_mk, rr_mk = pq_eval(p, X, Q, exact_top10)
            sweep.append({"m": m, "k": k, "bytes": b,
                          "recall_raw": raw_mk, "recall_rerank": rr_mk})
            log(f"  m={m:>2} k={k:>3}: {b:4.1f} bytes/vec  "
                f"recall raw={raw_mk:.3f}  re-ranked={rr_mk:.3f}")
    results["mk_sweep"] = sweep

    plt.figure(figsize=(8, 5))
    for row in sweep:
        plt.scatter(row["bytes"], row["recall_rerank"], s=60)
        plt.annotate(f"m={row['m']},k={row['k']}",
                     (row["bytes"], row["recall_rerank"]),
                     textcoords="offset points", xytext=(6, 4))
    plt.axhline(1.0, color="gray", ls="--", label="exact")
    plt.xlabel("PACKED bytes / vector  (m*log2(k)/8)")
    plt.ylabel(f"recall@10 after exact re-rank of top-{RERANK_DEPTH}")
    plt.title("PQ (m, k) sweep: memory vs recall")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(STORE_DIR, "pq_memory_sweep.png")
    plt.savefig(out)
    plt.close()
    log(f"plot saved -> {out}")

    banner("stage 7/8 -- IVF-PQ: skip cells (IVF), ADC-score survivors (PQ), "
           "exact re-rank the shortlist")
    tops, fracs = [], []
    for qi in range(len(Q)):
        cells = ivf.cell_order(Q[qi])[:knee]
        cand = np.concatenate([ivf.lists[c] for c in cells])
        lut = np.einsum("mkd,md->mk", pq.codebooks,
                        Q[qi].reshape(pq.m, pq.dsub))
        s = lut[np.arange(pq.m), pq.codes[cand]].sum(axis=1)
        short = cand[np.argsort(-s)[:RERANK_DEPTH]]
        s_exact = X[short] @ Q[qi]
        tops.append(short[np.argsort(-s_exact)[:TOP_K]])
        fracs.append(len(cand) / N)
    ivfpq_recall, _ = recall_at_10(tops, exact_top10)
    ivfpq_frac = float(np.mean(fracs))
    log(f"IVF-PQ (nprobe={knee}, m=8, k=256): recall@10={ivfpq_recall:.3f}, "
        f"fraction scanned={ivfpq_frac:.3f}, {pq_bytes:.0f} bytes/vec")
    results["ivfpq"] = {"nprobe": knee, "recall": ivfpq_recall,
                        "frac_scanned": ivfpq_frac, "bytes_per_vec": pq_bytes}

    banner("stage 8/8 -- the tradeoff table (recall / work / memory)")
    ivf_knee = next(r for r in ivf_rows if r["nprobe"] == knee)
    flat_bytes = DIM * 4
    table = [
        ("exact",                 1.000,               1.000,
         flat_bytes, 1.0),
        (f"IVF@knee (nprobe={knee})", ivf_knee["recall"], ivf_knee["frac_scanned"],
         flat_bytes, 1.0),
        ("PQ(8,256)+re-rank",     rr,                  1.000,
         pq_bytes, flat_bytes / pq_bytes),
        ("IVF-PQ+re-rank",        ivfpq_recall,        ivfpq_frac,
         pq_bytes, flat_bytes / pq_bytes),
    ]
    header = f"{'index':<24}{'recall@10':>10}{'frac scanned':>14}" \
             f"{'bytes/vec':>11}{'compression':>13}"
    print("\n" + header)
    print("-" * len(header))
    for name, rec, frac, b, comp in table:
        print(f"{name:<24}{rec:>10.3f}{frac:>14.3f}{b:>11.1f}{comp:>12.0f}x")
    results["tradeoff_table"] = [
        {"index": n, "recall": r, "frac_scanned": f,
         "bytes_per_vec": b, "compression": c}
        for n, r, f, b, c in table]

    out = os.path.join(STORE_DIR, "results_lab.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"numbers saved in -> {out}")


if __name__ == "__main__":
    main()
