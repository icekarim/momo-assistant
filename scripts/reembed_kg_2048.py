"""One-off migration: re-embed every KG entity at the new GEMINI_EMBEDDING_DIM.

Run once to bring all existing embeddings down from 3072 dim to 2048 dim so
they fit Firestore's flat vector index limit. After this completes successfully,
this script can be deleted (or kept for future dim changes).

Usage: python3 scripts/reembed_kg_2048.py
"""
import sys
import time
from pathlib import Path

# Make the repo root importable so we can use the existing helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from knowledge_graph import _build_embedding_text, _get_embedding
from conversation_store import get_db


def main():
    db = get_db()
    docs = list(db.collection(config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION).stream())
    target_dim = config.GEMINI_EMBEDDING_DIM
    total = len(docs)
    print(f"Re-embedding {total} entries to {target_dim} dim...", flush=True)

    updated, skipped, failed = 0, 0, 0
    started = time.time()

    for i, doc in enumerate(docs, 1):
        data = doc.to_dict()
        existing = data.get("embedding") or []
        if len(existing) == target_dim:
            skipped += 1
            if i % 100 == 0:
                print(f"  [{i}/{total}] +{updated} ~{skipped} !{failed}", flush=True)
            continue

        try:
            text = _build_embedding_text(data, source_type=data.get("source_type", ""))
            new_embedding = _get_embedding(text)
            if len(new_embedding) != target_dim:
                raise RuntimeError(
                    f"expected {target_dim} dim, got {len(new_embedding)}"
                )
            doc.reference.update({
                "embedding": new_embedding,
                "embedding_model": config.GEMINI_EMBEDDING_MODEL,
                "embedding_dim": target_dim,
            })
            updated += 1
        except Exception as exc:
            print(f"  ! {doc.id}: {exc}", flush=True)
            failed += 1

        time.sleep(0.1)

        if i % 100 == 0:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            eta = (total - i) / rate if rate else 0
            print(
                f"  [{i}/{total}] +{updated} ~{skipped} !{failed} "
                f"({rate:.1f}/s, eta {eta:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.0f}s: {updated} updated, {skipped} already-correct, {failed} failed",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
