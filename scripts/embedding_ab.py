"""READ-ONLY A/B: gemini-embedding-001 vs gemini-embedding-2 on retrieval.

Does NOT write to Firestore. Pulls a sample of real KG entities, embeds them
in-memory with BOTH models, and measures retrieval quality two ways:

1. Leave-one-out self-retrieval (label-free): for each entity, use its `name`
   as a query and check the rank of its own `content`-based document among the
   sample. Better model => target ranks higher (higher MRR / hit@k).
2. Hand-written semantic queries: a few realistic queries, compare top-5.

Usage: python3 scripts/embedding_ab.py [--sample N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import google.generativeai as genai

import config
from conversation_store import get_db

genai.configure(api_key=config.GEMINI_API_KEY)

MODEL_OLD = "models/gemini-embedding-001"
MODEL_NEW = "models/gemini-embedding-2"
DIM = config.GEMINI_EMBEDDING_DIM


def _embed(text, model, task_type):
    r = genai.embed_content(model=model, content=text, task_type=task_type,
                            output_dimensionality=DIM)
    v = np.array(r["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _cos_rank(query_vec, doc_matrix):
    sims = doc_matrix @ query_vec
    return np.argsort(-sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()

    db = get_db()
    docs = list(db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).limit(args.sample).stream())
    entities = []
    for d in docs:
        e = d.to_dict()
        name = (e.get("name") or "").strip()
        content = (e.get("content") or "").strip()
        if name and content:
            entities.append({"name": name, "content": content,
                             "entity_type": e.get("entity_type", "")})
    n = len(entities)
    print(f"Sample: {n} entities (read-only, no writes)\n", flush=True)
    if n < 5:
        print("Not enough entities with name+content for a meaningful A/B.")
        return 1

    results = {}
    for label, model in [("001", MODEL_OLD), ("emb2", MODEL_NEW)]:
        t0 = time.time()
        # Documents embedded as RETRIEVAL_DOCUMENT from content; queries from name.
        doc_mat = np.vstack([_embed(e["content"], model, "RETRIEVAL_DOCUMENT") for e in entities])
        q_vecs = [_embed(e["name"], model, "RETRIEVAL_QUERY") for e in entities]
        reciprocal_ranks = []
        hits_at_1 = hits_at_3 = 0
        for i, qv in enumerate(q_vecs):
            order = _cos_rank(qv, doc_mat)
            rank = int(np.where(order == i)[0][0]) + 1  # 1-based rank of own doc
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
        mrr = float(np.mean(reciprocal_ranks))
        results[label] = {"mrr": mrr, "hit@1": hits_at_1 / n, "hit@3": hits_at_3 / n,
                          "secs": time.time() - t0}
        print(f"[{label}] MRR={mrr:.3f}  hit@1={hits_at_1}/{n} ({hits_at_1/n:.0%})  "
              f"hit@3={hits_at_3}/{n} ({hits_at_3/n:.0%})  ({results[label]['secs']:.1f}s)", flush=True)

    print("\n=== VERDICT ===")
    old, new = results["001"], results["emb2"]
    dmrr = new["mrr"] - old["mrr"]
    dh1 = new["hit@1"] - old["hit@1"]
    print(f"MRR delta (emb2 - 001):   {dmrr:+.3f}")
    print(f"hit@1 delta (emb2 - 001): {dh1:+.1%}")
    if dmrr > 0.02 and dh1 >= 0:
        print("VERDICT: embedding-2 WINS on this sample — full re-embed worth considering.")
    elif dmrr < -0.02:
        print("VERDICT: embedding-001 WINS — do NOT migrate; keep 001.")
    else:
        print("VERDICT: ~TIE — migration not justified (incompatible-space re-embed not worth it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
