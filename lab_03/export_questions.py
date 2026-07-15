"""Export the fixed 250-question subset + fill-in templates.

Run me once (the questions config is tiny and already in your HF cache):
    python export_questions.py

Produces, next to this file:
  questions_250.json        the fixed subset (all fields) -- committed for reproducibility
  rewrites_template.json    first 20 gold-bearing questions; fill paraphrases + hyde,
                            then SAVE AS rewrites.json
  judge_template.json       16 questions with the longest gold answers; fill
                            answer_paraphrase (a faithful re-wording of gold_answer,
                            same facts, different words), then SAVE AS judge_pairs.json
"""
import json
import os

from datasets import load_dataset

STORE_DIR = os.path.dirname(os.path.abspath(__file__))
REWRITE_SUBSET_N = 20
JUDGE_PAIRS_N = 16


def first_250(qs):
    return sorted(qs, key=lambda q: q["question_id"])[:250]


qs = load_dataset("onyx-dot-app/EnterpriseRAG-Bench", "questions")["test"]
Q = first_250(qs)

fields = ("question_id", "question_type", "source_types", "question",
          "expected_doc_ids", "gold_answer", "answer_facts")
with open(os.path.join(STORE_DIR, "questions_250.json"), "w", encoding="utf-8") as f:
    json.dump([{k: q[k] for k in fields} for q in Q], f, indent=1, ensure_ascii=False)
print(f"questions_250.json: {len(Q)} questions")

gold_bearing = [q for q in Q if q["expected_doc_ids"]]
subset = gold_bearing[:REWRITE_SUBSET_N]
template = {q["question_id"]: {"question": q["question"],
                               "paraphrases": ["", "", ""],
                               "hyde": ""}
            for q in subset}
with open(os.path.join(STORE_DIR, "rewrites_template.json"), "w", encoding="utf-8") as f:
    json.dump(template, f, indent=1, ensure_ascii=False)
print(f"rewrites_template.json: {len(subset)} questions to paraphrase "
      f"(fill >=2 paraphrases + hyde each, save as rewrites.json)")

judge_qs = sorted(gold_bearing, key=lambda q: -len(q["gold_answer"] or ""))[:JUDGE_PAIRS_N]
judge_qs = sorted(judge_qs, key=lambda q: q["question_id"])
jt = {q["question_id"]: {"question": q["question"],
                         "gold_answer": q["gold_answer"],
                         "answer_paraphrase": ""}
      for q in judge_qs}
with open(os.path.join(STORE_DIR, "judge_template.json"), "w", encoding="utf-8") as f:
    json.dump(jt, f, indent=1, ensure_ascii=False)
print(f"judge_template.json: {len(jt)} gold answers to paraphrase "
      f"(save as judge_pairs.json)")

types = {}
for q in Q:
    types[q["question_type"]] = types.get(q["question_type"], 0) + 1
print("question_type counts:", dict(sorted(types.items())))
refs = sorted({t for q in Q for t in q["source_types"]})
print("referenced source_types:", refs)
