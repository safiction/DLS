"""
Inputs expected next to this file (see export_questions.py):
  questions_250.json, rewrites.json, judge_pairs.json
Heavy steps cache themselves (corpus jsonl, chunk/fact embeddings npy).
"""

import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset   # EnterpriseRAG-Bench -- MIT
from sentence_transformers import SentenceTransformer
from scipy.stats import ttest_rel, ttest_ind, wilcoxon, t, sem

SEED = 20260605
CAP_PER_TYPE = 1000        # seeded sample cap per referenced source_type slice
MATCH_TH = 0.6             # THE match rule: fact present iff cos(fact, chunk) >= 0.6
ABSTAIN_TAU = 0.35         # abstention threshold on top-1 chunk similarity
TOP_K_CHUNKS = 20          # retrieved context = top-20 chunks
K_EVAL = 10                # doc-level recall@k / nDCG@k cutoff
RANK_DEPTH = 500           # chunk-ranking depth kept per variant
CHUNK_SIZES = [100, 200, 400]      # in whitespace tokens (stated tokenization)
OVERLAPS = [0, 25, 50]
RRF_K = 60
JUDGE_MODEL = "llama3.1:8b"        # ollama pull llama3.1:8b
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
STORE_DIR = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(msg):
    print("\n" + "=" * 72, flush=True)
    print(msg, flush=True)
    print("=" * 72, flush=True)


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def first_250(qs):
    return sorted(qs, key=lambda q: q["question_id"])[:250]

def load_questions():
    path = os.path.join(STORE_DIR, "questions_250.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            Q = json.load(f)
        log(f"questions_250.json: {len(Q)} questions (committed fixed subset)")
        return Q
    qs = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")["test"]
    Q = [{k: q[k] for k in ("question_id", "question_type", "source_types",
                            "question", "expected_doc_ids", "gold_answer",
                            "answer_facts")} for q in first_250(qs)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(Q, f, indent=1, ensure_ascii=False)
    return Q


def build_corpus(questions):
    referenced = {s.lower() for q in questions for s in q["source_types"]}
    gold_ids = {i for q in questions for i in q["expected_doc_ids"]}
    log(f"referenced source_types: {sorted(referenced)}")
    log(f"{len(gold_ids)} unique gold doc_ids -- never dropped")

    cache = os.path.join(STORE_DIR, "assignment_corpus.jsonl")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]
        log(f"corpus cache hit: {len(corpus)} docs")
        return corpus

    log("corpus cache miss -> streaming ~500K docs (one-time, slow)")
    stream = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "documents",
                          split="test", streaming=True)
    rng = np.random.default_rng(SEED)
    gold, reservoirs, seen_per_type = {}, defaultdict(list), defaultdict(int)
    t0 = time.time()
    for n, d in enumerate(stream, 1):
        st = d["source_type"].lower()
        if d["doc_id"] in gold_ids:
            gold[d["doc_id"]] = d
        elif st in referenced:
            seen_per_type[st] += 1
            res = reservoirs[st]
            if len(res) < CAP_PER_TYPE:
                res.append(d)
            else:
                j = int(rng.integers(0, seen_per_type[st]))
                if j < CAP_PER_TYPE:
                    res[j] = d
        if n % 25000 == 0:
            log(f"  streamed {n:>7,} | gold {len(gold)}/{len(gold_ids)} | "
                + " ".join(f"{k}:{len(v)}" for k, v in sorted(reservoirs.items()))
                + f" | {time.time()-t0:,.0f}s")
    log(f"stream done: {n:,} docs in {time.time()-t0:,.0f}s")
    missing = gold_ids - set(gold)
    if missing:
        log(f"WARNING: {len(missing)} gold docs never appeared: {sorted(missing)[:5]}")

    corpus = list(gold.values())
    for st in sorted(reservoirs):
        corpus += [d for d in reservoirs[st] if d["doc_id"] not in gold]
    fields = ("doc_id", "source_type", "title", "content")
    with open(cache, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps({k: d.get(k) for k in fields},
                               ensure_ascii=False) + "\n")
    log(f"corpus = {len(gold)} gold + sampled slices = {len(corpus):,} docs -> cached")
    return corpus

def chunk_fixed(text, size, overlap):
    assert 0 <= overlap < size
    toks = text.split()
    if not toks:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(toks), step):
        chunks.append(" ".join(toks[start:start + size]))
        if start + size >= len(toks):
            break
    return chunks


def _split_pieces(text, size, level=0):
    """paragraph -> sentence -> word: break any piece longer than `size`."""
    if len(text.split()) <= size:
        return [text]
    if level == 0:
        parts = [p for p in text.split("\n\n") if p.strip()]
        if len(parts) > 1:
            return [x for p in parts for x in _split_pieces(p, size, 1)]
        level = 1
    if level == 1:
        parts = [p for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
        if len(parts) > 1:
            return [x for p in parts for x in _split_pieces(p, size, 2)]
    toks = text.split()
    return [" ".join(toks[i:i + size]) for i in range(0, len(toks), size)]


def chunk_recursive(text, size):
    """Recursive splitting, then greedy merge of consecutive pieces up to `size`."""
    chunks, buf, blen = [], [], 0
    for p in _split_pieces(text, size):
        n = len(p.split())
        if blen + n > size and buf:
            chunks.append(" ".join(buf))
            buf, blen = [], 0
        buf.append(p)
        blen += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def chunking_study(questions, doc_by_id, model):
    """Recall floor per config: fraction of answer_facts present (cos >= 0.6)
    in SOME chunk of the fact's expected docs. Cached to study_cache.json."""
    cache = os.path.join(STORE_DIR, "study_cache.json")
    eval_qs = [q for q in questions
               if q["answer_facts"] and any(d in doc_by_id for d in q["expected_doc_ids"])]
    facts = [(q["question_id"], fi, fact, [d for d in q["expected_doc_ids"]
                                           if d in doc_by_id])
             for q in eval_qs for fi, fact in enumerate(q["answer_facts"])]
    log(f"chunking study: {len(eval_qs)} questions, {len(facts)} answer_facts")

    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            study = json.load(f)
        log("study cache hit")
        return study

    fact_emb = normalize(model.encode([f[2] for f in facts], batch_size=64,
                                      convert_to_numpy=True).astype(np.float32))
    gold_docs = sorted({d for f in facts for d in f[3]})
    configs = ([("fixed", s, o) for s in CHUNK_SIZES for o in OVERLAPS]
               + [("recursive", s, 0) for s in CHUNK_SIZES])
    study = {"configs": [], "presence": {}, "exhibit": None, "formula_check": None}

    for kind, size, ov in configs:
        key = f"{kind}_{size}_{ov}"
        texts, owner = [], []                        # owner[i] = doc_id of chunk i
        for did in gold_docs:
            cs = (chunk_fixed(doc_by_id[did].get("content") or "", size, ov)
                  if kind == "fixed"
                  else chunk_recursive(doc_by_id[did].get("content") or "", size))
            texts += cs
            owner += [did] * len(cs)
        emb = normalize(model.encode(texts, batch_size=64, show_progress_bar=False,
                                     convert_to_numpy=True).astype(np.float32))
        rows_of = defaultdict(list)
        for i, did in enumerate(owner):
            rows_of[did].append(i)
        present = []
        for fi, (_, _, _, docs) in enumerate(facts):
            rows = [r for d in docs for r in rows_of[d]]
            best = float((emb[rows] @ fact_emb[fi]).max()) if rows else 0.0
            present.append(1 if best >= MATCH_TH else 0)
        floor = sum(present) / len(present)
        study["configs"].append({"kind": kind, "size": size, "overlap": ov,
                                 "floor": floor, "n_chunks": len(texts)})
        study["presence"][key] = present
        log(f"  {kind:>9} size={size:>3} overlap={ov:>2}: "
            f"floor={floor:.3f}  ({len(texts):,} chunks over gold docs)")

    # boundary-split exhibit: present at the largest overlap, absent at overlap 0
    ov_hi = OVERLAPS[-1]
    for s in CHUNK_SIZES:
        p0 = study["presence"][f"fixed_{s}_{OVERLAPS[0]}"]
        p50 = study["presence"][f"fixed_{s}_{ov_hi}"]
        for fi in range(len(facts)):
            if p0[fi] == 0 and p50[fi] == 1:
                qid, _, fact, docs = facts[fi]
                study["exhibit"] = {"question_id": qid, "fact": fact,
                                    "doc_id": docs[0], "size": s,
                                    "overlap": ov_hi}
                break
        if study["exhibit"]:
            break
    if study["exhibit"]:
        e = study["exhibit"]
        print(f"\n--- boundary-split exhibit (fixed, size={e['size']}) ---\n"
              f"fact ({e['question_id']}): {e['fact']}\n"
              f"present=0 at overlap 0 (split across a chunk boundary), "
              f"present=1 at overlap {e['overlap']} -- a small overlap rescues it\n",
              flush=True)

    # chunk-count formula check on the longest gold doc
    did = max(gold_docs, key=lambda d: len((doc_by_id[d].get("content") or "").split()))
    L = len((doc_by_id[did].get("content") or "").split())
    s, o = CHUNK_SIZES[len(CHUNK_SIZES) // 2], OVERLAPS[-1]
    actual = len(chunk_fixed(doc_by_id[did].get("content") or "", s, o))
    formula = 1 if L <= s else math.ceil((L - o) / (s - o))
    study["formula_check"] = {"doc_id": did, "L": L, "size": s, "overlap": o,
                              "actual": actual, "formula": formula}
    log(f"formula check: L={L}, size={s}, ov={o} -> "
        f"actual {actual} vs ceil((L-o)/(s-o)) = {formula}")

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(study, f)
    return study


def plot_study(study):
    plt.figure(figsize=(8, 5))
    for ov in OVERLAPS:
        pts = [(c["size"], c["floor"]) for c in study["configs"]
               if c["kind"] == "fixed" and c["overlap"] == ov]
        plt.plot(*zip(*pts), marker="o", label=f"fixed, overlap={ov}")
    pts = [(c["size"], c["floor"]) for c in study["configs"]
           if c["kind"] == "recursive"]
    plt.plot(*zip(*pts), marker="s", ls="--", label="recursive")
    plt.xlabel("chunk size (whitespace tokens)")
    plt.ylabel(f"recall floor (fact containment, cos >= {MATCH_TH})")
    plt.title("Chunking study: the recall floor")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(STORE_DIR, "chunking_recall_floor.png")
    plt.savefig(out)
    plt.close()
    log(f"plot saved -> {out}")


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def build_inverted_index(texts):
    index = defaultdict(list)
    lengths = np.zeros(len(texts), dtype=np.int32)
    for cid, text in enumerate(texts):
        toks = tokenize(text)
        lengths[cid] = len(toks)
        tf = defaultdict(int)
        for tok in toks:
            tf[tok] += 1
        for term, freq in tf.items():
            index[term].append((cid, freq))
    avgdl = float(lengths.mean())
    return index, lengths, avgdl


def bm25_search(query, index, lengths, avgdl, N, top_k=RANK_DEPTH, k1=1.5, b=0.75):
    scores = defaultdict(float)
    for term in set(tokenize(query)):
        postings = index.get(term)
        if not postings:
            continue
        df = len(postings)
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for cid, tf in postings:
            denom = k1 * ((1 - b) + b * lengths[cid] / avgdl) + tf
            scores[cid] += idf * (k1 + 1) * tf / denom
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [cid for cid, _ in ranked]


def build_chunk_corpus(corpus, cfg, model):
    """Chunk every doc at the chosen config; cache chunks + dense embeddings."""
    kind, size, ov = cfg
    tag = f"{kind}_{size}_{ov}"
    cpath = os.path.join(STORE_DIR, f"chunks_{tag}.jsonl")
    epath = os.path.join(STORE_DIR, f"emb_chunks_{tag}.npy")
    if os.path.exists(cpath) and os.path.exists(epath):
        with open(cpath, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
        emb = np.load(epath)
        if len(recs) == len(emb):
            log(f"chunk cache hit: {len(recs):,} chunks ({tag})")
            return [r["text"] for r in recs], [r["doc_id"] for r in recs], emb
    texts, parents = [], []
    for d in corpus:
        content = d.get("content") or ""
        cs = (chunk_fixed(content, size, ov) if kind == "fixed"
              else chunk_recursive(content, size))
        texts += cs
        parents += [d["doc_id"]] * len(cs)
    log(f"chunked corpus at {tag}: {len(texts):,} chunks; embedding "
        "(one-time, cached)...")
    emb = normalize(model.encode(texts, batch_size=64, show_progress_bar=True,
                                 convert_to_numpy=True).astype(np.float32))
    with open(cpath, "w", encoding="utf-8") as f:
        for tx, p in zip(texts, parents):
            f.write(json.dumps({"doc_id": p, "text": tx}, ensure_ascii=False) + "\n")
    np.save(epath, emb)
    return texts, parents, emb


def dedup_to_docs(chunk_ranking, parents):
    """Collapse a chunk ranking to the first-seen rank of each distinct doc."""
    seen, docs = set(), []
    for cid in chunk_ranking:
        d = parents[cid]
        if d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def recall_at_k(docs, gold, k=K_EVAL):
    return len(set(docs[:k]) & gold) / len(gold)

def mrr(docs, gold):
    for i, d in enumerate(docs, 1):
        if d in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(docs, gold, k=K_EVAL):
    """Binary relevance: gains 0/1 (2^rel - 1), log2(i+1) discount."""
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(docs[:k], 1) if d in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0

def context_precision(chunk_ranking, parents, gold, k=K_EVAL):
    """MAP with chunks DEDUPED to parent docs first -- otherwise near-duplicate
    chunks of one gold doc pin precision near 1 and drift with chunk size."""
    docs = dedup_to_docs(chunk_ranking, parents)[:k]
    hits, ap = 0, 0.0
    for i, d in enumerate(docs, 1):
        if d in gold:
            hits += 1
            ap += hits / i
    return ap / len(gold) if gold else 0.0


def context_recall(chunk_ranking, chunk_emb, fact_emb):
    """(# answer_facts present in the retrieved context) / (# facts)"""
    if fact_emb is None or not len(fact_emb):
        return None
    ctx = chunk_ranking[:TOP_K_CHUNKS]
    sims = fact_emb @ chunk_emb[ctx].T
    return float((sims.max(axis=1) >= MATCH_TH).mean())


def load_rewrites():
    path = os.path.join(STORE_DIR, "rewrites.json")
    if not os.path.exists(path):
        log("rewrites.json NOT FOUND -- run export_questions.py, fill the "
            "template, save as rewrites.json. Skipping rewrite experiments.")
        return {}
    with open(path, encoding="utf-8") as f:
        rw = json.load(f)
    rw = {qid: {"paraphrases": [p for p in v.get("paraphrases", []) if p.strip()],
                "hyde": v.get("hyde", "").strip()}
          for qid, v in rw.items()}
    rw = {qid: v for qid, v in rw.items() if v["paraphrases"] and v["hyde"]}
    log(f"rewrites.json: {len(rw)} questions with paraphrases + hyde")
    return rw


def rrf_merge(rankings, n_chunks):
    """RAG-Fusion by hand: RRF over the paraphrase runs, k=60."""
    score = np.zeros(n_chunks)
    for r in rankings:
        for rank, cid in enumerate(r, 1):
            score[cid] += 1.0 / (RRF_K + rank)
    return list(np.argsort(-score)[:RANK_DEPTH])


def run_variants(q, rw, dense_score_fn, bm25_fn, n_chunks):
    """Return {variant: chunk_ranking}. Multi-query merges runs by the MAX
    similarity a chunk got across paraphrase runs (a set-union is not a ranking)."""
    out = {}
    s_plain = dense_score_fn(q["question"])
    out["bm25_plain"] = bm25_fn(q["question"])
    out["dense_plain"] = list(np.argsort(-s_plain)[:RANK_DEPTH])
    if rw:
        runs = [s_plain] + [dense_score_fn(p) for p in rw["paraphrases"]]
        out["multi_query"] = list(np.argsort(-np.max(np.stack(runs), 0))[:RANK_DEPTH])
        out["hyde"] = list(np.argsort(-dense_score_fn(rw["hyde"]))[:RANK_DEPTH])
        out["rag_fusion"] = rrf_merge(
            [list(np.argsort(-s)[:RANK_DEPTH]) for s in runs], n_chunks)
    return out


def ollama_chat(prompt, cache):
    import requests
    key = hashlib.md5((JUDGE_MODEL + prompt).encode()).hexdigest()
    if key in cache:
        return cache[key]
    r = requests.post(OLLAMA_URL, json={
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0, "seed": SEED},
        "stream": False,
    }, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    cache[key] = content
    return content


JUDGE_PROMPT = """You are an impartial judge. Two answers to the same question follow.
Score EACH answer from 1 to 5 on: relevance (addresses the question),
grounding (consistent, no invented specifics), completeness (covers the needed facts).
Then pick the better answer overall.
Respond with ONLY this JSON, nothing else:
{{"A": {{"relevance": n, "grounding": n, "completeness": n}},
  "B": {{"relevance": n, "grounding": n, "completeness": n}}, "winner": "A" or "B" or "tie"}}

Question: {question}

Answer A: {a}

Answer B: {b}"""


def judge_pair(question, first, second, cache):
    raw = ollama_chat(JUDGE_PROMPT.format(question=question, a=first, b=second),
                      cache)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        v = json.loads(m.group(0))
        return v.get("winner", "tie"), v
    except (AttributeError, json.JSONDecodeError):
        return "tie", None


def pad_answer(ans):
    """On-topic filler that adds NO new fact (verbosity-bias probe)."""
    first = (ans.split(". ")[0]).strip()
    return (ans + " To recap the main point: " + first + "."
            " As noted above, this directly addresses what was asked."
            " In summary, the details provided here cover the relevant"
            " information for this question without omitting anything essential;"
            " it is worth restating that the core answer remains as described.")


def judge_bias_lab(questions):
    banner("stage 6/8 -- LLM-judge bias test (Ollama, temperature 0, fixed seed)")
    # The Goodhart weight-flip is PURELY BY HAND -- no model needed:
    A, C = [5, 5, 3], [4, 3, 5]
    wa, wc = (A[0] + A[1] + 2 * A[2]) / 4, (C[0] + C[1] + 2 * C[2]) / 4
    print(f"""
--- Goodhart weight-flip (by hand) ---
honest A = mean{A} = {np.mean(A):.4f}   beats   padded C = mean{C} = {np.mean(C):.4f}
re-weight completeness x2:
  A = (5+5+2*3)/4 = {wa:.2f}   loses to   C = (4+3+2*5)/4 = {wc:.2f}
no word changed -- the rubric weights are the attack surface
""", flush=True)

    path = os.path.join(STORE_DIR, "judge_pairs.json")
    if not os.path.exists(path):
        log("judge_pairs.json NOT FOUND -- fill judge_template.json and save "
            "as judge_pairs.json. Skipping the empirical bias rates.")
        return None
    with open(path, encoding="utf-8") as f:
        pairs = {k: v for k, v in json.load(f).items()
                 if v.get("answer_paraphrase", "").strip()}
    by_id = {q["question_id"]: q for q in questions}
    log(f"{len(pairs)} answer pairs; judge = {JUDGE_MODEL}")

    cpath = os.path.join(STORE_DIR, "judge_cache.json")
    cache = {}
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            cache = json.load(f)
    try:
        pos_trials, verb_trials, per_pair = [], [], {}
        for n, (qid, v) in enumerate(sorted(pairs.items()), 1):
            gold = by_id[qid]["gold_answer"]
            para, padded, qtext = v["answer_paraphrase"], pad_answer(gold), v["question"]
            # position: a genuine tie (gold vs its paraphrase), BOTH orderings
            w1, _ = judge_pair(qtext, gold, para, cache)      # gold first
            w2, _ = judge_pair(qtext, para, gold, cache)      # paraphrase first
            pos_trials += [w1 == "A", w2 == "A"]              # first slot wins?
            per_pair[qid] = (w1, w2)
            # verbosity: same content, longer B -- both orderings
            v1, _ = judge_pair(qtext, gold, padded, cache)    # padded is B
            v2, _ = judge_pair(qtext, padded, gold, cache)    # padded is A
            verb_trials += [v1 == "B", v2 == "A"]             # longer wins?
            log(f"  pair {n}/{len(pairs)} ({qid}): tie-orders=({w1},{w2}) "
                f"verbose-orders=({v1},{v2})")
        pos_rate = float(np.mean(pos_trials))
        verb_rate = float(np.mean(verb_trials))
        # swap-and-average: count a win only if the same answer wins both orders
        # ("A","B") -> gold won both orderings; ("B","A") -> paraphrase won both
        decisive = sum(1 for w in per_pair.values() if w in (("A", "B"), ("B", "A")))
        residual = decisive / len(per_pair)
        print(f"""
--- empirical bias rates ({JUDGE_MODEL}, temperature 0) ---
position-follow rate (ties): {pos_rate:.2f}  (first slot wins; 0.5 = unbiased on ties)
verbosity: longer answer wins {verb_rate:.2f} of trials (content identical)
swap-and-average: {residual:.2f} of tie pairs stay decided (same winner both
orders) -- position-driven flips are removed; reduction vs raw = {pos_rate - residual:.2f}
""", flush=True)
        return {"position_follow_rate": pos_rate, "verbosity_win_rate": verb_rate,
                "swap_avg_decisive_rate": residual, "n_pairs": len(per_pair)}
    except Exception as exc:                     # ollama down / model missing
        log(f"judge stage skipped ({type(exc).__name__}: {exc}) -- start Ollama "
            f"and `ollama pull {JUDGE_MODEL}`, then re-run (responses are cached)")
        return None
    finally:
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(cache, f)

def significance(diff_pairs, label_a, label_b):
    """Paired test on per-query context precision (skeleton pattern:
    paired vs Welch; the paired one is the honest choice here)."""
    a, b = np.array([x for x, _ in diff_pairs]), np.array([y for _, y in diff_pairs])
    d = b - a
    t_rel, p_rel = ttest_rel(b, a)
    t_ind, p_ind = ttest_ind(b, a, equal_var=False)      # Welch, for contrast
    try:
        w_stat, p_w = wilcoxon(d)
    except ValueError:                                    # all differences zero
        w_stat, p_w = float("nan"), 1.0
    lo, hi = t.interval(0.95, len(d) - 1, loc=d.mean(), scale=sem(d))
    print(f"""
--- significance: {label_b} vs {label_a} (n={len(d)} paired queries) ---
mean context precision: {a.mean():.4f} -> {b.mean():.4f}   mean diff = {d.mean():+.4f}
95% CI for the diff:    [{lo:+.4f}, {hi:+.4f}]
paired t-test:  t = {t_rel:+.3f}, p = {p_rel:.4f}
Wilcoxon:       W = {w_stat}, p = {p_w:.4f}
(Welch, for contrast -- ignores pairing: t = {t_ind:+.3f}, p = {p_ind:.4f})
caveats: significance is necessary-not-sufficient; after sweeping many
configs the winning p-value is subject to multiple comparisons
""", flush=True)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "t_paired": float(t_rel), "p_paired": float(p_rel),
            "wilcoxon_p": float(p_w), "n": len(d)}

def main():
    results = {"seed": SEED, "match_threshold": MATCH_TH, "tau": ABSTAIN_TAU}

    banner("stage 1/8 -- fixed question subset + streamed corpus")
    questions = load_questions()
    corpus = build_corpus(questions)
    doc_by_id = {d["doc_id"]: d for d in corpus}
    results["n_docs"] = len(corpus)

    log("loading sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    banner("stage 2/8 -- chunking study: the recall floor")
    study = chunking_study(questions, doc_by_id, model)
    plot_study(study)
    best = max(study["configs"], key=lambda c: (c["floor"], -c["n_chunks"]))
    log(f"RECALL FLOOR at best config ({best['kind']}, size={best['size']}, "
        f"overlap={best['overlap']}): {best['floor']:.3f} -- the best any "
        "generator downstream could do")
    results["chunking"] = study["configs"]
    results["recall_floor"] = best["floor"]
    results["exhibit"] = study["exhibit"]

    banner("stage 3/8 -- chunk index at the chosen config "
           f"({best['kind']}, {best['size']}, {best['overlap']})")
    cfg = (best["kind"], best["size"], best["overlap"])
    chunk_texts, parents, chunk_emb = build_chunk_corpus(corpus, cfg, model)
    n_chunks = len(chunk_texts)
    index, lengths, avgdl = build_inverted_index(chunk_texts)
    log(f"BM25 re-indexed on {n_chunks:,} chunks "
        f"(vocab {len(index):,}, avgdl {avgdl:.0f})")

    def dense_score_fn(text):
        e = normalize(model.encode([text], convert_to_numpy=True).astype(np.float32))
        return (chunk_emb @ e[0])

    def bm25_fn(text):
        return bm25_search(text, index, lengths, avgdl, n_chunks)

    banner("stage 4/8 -- query rewriting: multi-query, HyDE, RAG-Fusion")
    rewrites = load_rewrites()
    eval_qs = [q for q in questions if q["expected_doc_ids"]]
    fact_embs = {}
    for q in eval_qs:
        if q["answer_facts"]:
            fact_embs[q["question_id"]] = normalize(model.encode(
                q["answer_facts"], convert_to_numpy=True).astype(np.float32))

    per_variant = defaultdict(dict)      # variant -> qid -> metrics
    rankings_cache = {}                  # (qid, variant) -> chunk ranking
    for n, q in enumerate(eval_qs, 1):
        qid = q["question_id"]
        gold = set(q["expected_doc_ids"])
        variants = run_variants(q, rewrites.get(qid), dense_score_fn, bm25_fn,
                                n_chunks)
        for name, ranking in variants.items():
            rankings_cache[(qid, name)] = ranking
            docs = dedup_to_docs(ranking, parents)
            per_variant[name][qid] = {
                "recall": recall_at_k(docs, gold),
                "mrr": mrr(docs, gold),
                "ndcg": ndcg_at_k(docs, gold),
                "cprec": context_precision(ranking, parents, gold),
                "crec": context_recall(ranking, chunk_emb,
                                       fact_embs.get(qid)),
                "qtype": q["question_type"],
            }
        if n % 25 == 0:
            log(f"  retrieved {n}/{len(eval_qs)} questions "
                f"({len(variants)} variants each)")

    order = ["bm25_plain", "dense_plain", "multi_query", "hyde", "rag_fusion"]
    print(f"\n{'variant':<14}{'recall@10':>10}{'MRR':>8}{'nDCG@10':>9}   "
          f"(plain retrievers, ALL {len(eval_qs)} gold-bearing questions)")
    for name in ("bm25_plain", "dense_plain"):
        m = list(per_variant[name].values())
        print(f"{name:<14}{np.mean([x['recall'] for x in m]):>10.3f}"
              f"{np.mean([x['mrr'] for x in m]):>8.3f}"
              f"{np.mean([x['ndcg'] for x in m]):>9.3f}")

    rw_qids = sorted({qid for (qid, v) in rankings_cache if v == "multi_query"})
    log(f"rewrite subset: {len(rw_qids)} questions (hand-written, committed)")
    print(f"\n{'variant':<14}{'recall@10':>10}{'MRR':>8}{'nDCG@10':>9}   "
          f"(on the {len(rw_qids)}-question rewrite subset)")
    for name in order:
        m = [per_variant[name][qid] for qid in rw_qids if qid in per_variant[name]]
        if not m:
            continue
        print(f"{name:<14}{np.mean([x['recall'] for x in m]):>10.3f}"
              f"{np.mean([x['mrr'] for x in m]):>8.3f}"
              f"{np.mean([x['ndcg'] for x in m]):>9.3f}")
    results["rewrite_table"] = {
        name: {k: float(np.mean([x[k] for x in
                                 [per_variant[name][qid] for qid in rw_qids
                                  if qid in per_variant[name]]]))
               for k in ("recall", "mrr", "ndcg")}
        for name in order if any(qid in per_variant[name] for qid in rw_qids)}

    banner("stage 5/8 -- RAG evaluation by hand, sliced by question_type")
    qtypes = sorted({q["question_type"] for q in eval_qs})
    for name in ("bm25_plain", "dense_plain"):
        print(f"\n{name}: context precision / context recall per question_type")
        rep = {}
        for qt in qtypes:
            m = [v for v in per_variant[name].values() if v["qtype"] == qt]
            cr = [v["crec"] for v in m if v["crec"] is not None]
            if m:
                rep[qt] = {"cprec": float(np.mean([v["cprec"] for v in m])),
                           "crec": float(np.mean(cr)) if cr else None,
                           "n": len(m)}
                crs = f"{rep[qt]['crec']:.3f}" if cr else "  n/a"
                print(f"  {qt:<28} n={len(m):>3}  cprec={rep[qt]['cprec']:.3f}  "
                      f"crec={crs}")
        results[f"rag_metrics_{name}"] = rep

    # abstention for info_not_found (metrics are N/A there: 0/0)
    inf = [q for q in questions if q["question_type"] == "info_not_found"]
    control = [q for q in questions
               if q["question_type"] == "basic" and q["expected_doc_ids"]]
    if not control:                       # fallback: most common gold-bearing type
        counts = defaultdict(int)
        for q in eval_qs:
            counts[q["question_type"]] += 1
        ctl_type = max(counts, key=counts.get)
        control = [q for q in eval_qs if q["question_type"] == ctl_type]
        log(f"no 'basic' questions in subset -> control category = {ctl_type}")
    def top1_sim(q):
        return float(dense_score_fn(q["question"]).max())
    if inf:
        sims_inf = [top1_sim(q) for q in inf]
        sims_ctl = [top1_sim(q) for q in control]
        abst = float(np.mean([s < ABSTAIN_TAU for s in sims_inf]))
        false_abst = float(np.mean([s < ABSTAIN_TAU for s in sims_ctl]))
        print(f"\nabstention (tau={ABSTAIN_TAU}): info_not_found correctly below "
              f"tau: {abst:.2f} (n={len(inf)}); control ('basic') falsely below: "
              f"{false_abst:.2f} (n={len(control)})")
        results["abstention"] = {"tau": ABSTAIN_TAU, "info_not_found": abst,
                                 "control_basic": false_abst}
        plt.figure(figsize=(8, 5))
        plt.hist(sims_inf, bins=20, alpha=0.6, label="info_not_found")
        plt.hist(sims_ctl, bins=20, alpha=0.6, label="basic (control)")
        plt.axvline(ABSTAIN_TAU, color="gray", ls="--", label=f"tau={ABSTAIN_TAU}")
        plt.xlabel("top-1 chunk cosine similarity")
        plt.ylabel("questions")
        plt.title("Abstention: top-1 similarity by category")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(STORE_DIR, "abstention_hist.png")
        plt.savefig(out)
        plt.close()
        log(f"plot saved -> {out}")

    results["judge"] = judge_bias_lab(questions)

    banner("stage 7/8 -- significance: best rewrite vs the BM25 baseline")
    if rewrites:
        best_rw = max(("multi_query", "hyde", "rag_fusion"),
                      key=lambda v: results["rewrite_table"].get(v, {}).get("ndcg", 0))
        pairs = [(per_variant["bm25_plain"][qid]["cprec"],
                  per_variant[best_rw][qid]["cprec"])
                 for qid in rw_qids if qid in per_variant[best_rw]]
        results["significance"] = significance(pairs, "bm25_plain", best_rw)
        results["best_rewrite"] = best_rw
    else:
        log("skipped (no rewrites.json yet)")

    banner("stage 8/8 -- save results")
    out = os.path.join(STORE_DIR, "results_assignment.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"all numbers saved -> {out}")


if __name__ == "__main__":
    main()
