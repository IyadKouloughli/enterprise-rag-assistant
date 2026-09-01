"""
build_faiss_index.py

Embeds the chunked JSONL produced by ingest_gitlab_handbook.py using a free
local sentence-transformers model, builds a FAISS index, and gives you a
query function that respects document-level ACLs (department + allowed_roles)
before returning results — the core piece your "authorization-aware
retrieval" resume bullet is about.

SETUP (one time)
    pip install sentence-transformers faiss-cpu numpy

USAGE — build the index
    python build_faiss_index.py build --chunks data/handbook_chunks.jsonl --out data/index

USAGE — query it
    python build_faiss_index.py query --index data/index --q "what is our vacation policy" --role hr
    python build_faiss_index.py query --index data/index --q "what is our vacation policy" --role engineer

Notice how the same query returns different (or zero) results depending on
--role once you tag some chunks as hr-only — that's your ACL demo.

FILES WRITTEN BY `build`
    data/index/embeddings.faiss   -- the FAISS vector index
    data/index/metadata.jsonl     -- parallel metadata (one line per vector,
                                      same order as the FAISS index)

NEXT STEPS AFTER THIS WORKS
    - Wrap `search()` in a FastAPI endpoint (POST /api/chat) — see the
      "Production API" section of your original project plan.
    - Add hybrid search: combine this vector score with a simple BM25/keyword
      score (rank_bm25 package) and merge with reciprocal rank fusion.
    - Swap SentenceTransformer for Azure OpenAI embeddings when you move to
      cloud deployment -- everything else (FAISS -> Azure AI Search, the ACL
      filter logic) stays conceptually the same.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small, fast, strong for its size, free
EMBED_DIM = 384


def load_chunks(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_index(chunks_path: Path, out_dir: Path, batch_size: int = 64):
    from sentence_transformers import SentenceTransformer
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_chunks(chunks_path)
    print(f"Loaded {len(records)} chunks from {chunks_path}")

    print(f"Loading embedding model: {MODEL_NAME} (first run downloads it, ~130MB)")
    model = SentenceTransformer(MODEL_NAME)

    texts = [r["text"] for r in records]
    print("Embedding chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use inner product = cosine similarity
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity via normalized IP
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "embeddings.faiss"))

    with open(out_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nIndex built: {index.ntotal} vectors, dim={embeddings.shape[1]}")
    print(f"Written to {out_dir}/embeddings.faiss and {out_dir}/metadata.jsonl")


def load_index(index_dir: Path):
    import faiss

    index = faiss.read_index(str(index_dir / "embeddings.faiss"))
    metadata = load_chunks(index_dir / "metadata.jsonl")
    return index, metadata


def search(index_dir: Path, query: str, role: str, top_k: int = 5, fetch_k: int = 30):
    """
    Retrieve top_k chunks the given role is authorized to see.

    ACL logic: over-fetch fetch_k candidates by similarity, filter to ones
    where `role` is in the chunk's allowed_roles, then take the top_k of
    what's left. This mirrors how you'd do post-filtering with a real vector
    DB when the DB itself doesn't support metadata filtering natively; Azure
    AI Search / Bedrock Knowledge Bases can push this filter down into the
    search call itself instead, which you should mention when you get to
    the cloud-deployment stage.
    """
    from sentence_transformers import SentenceTransformer

    index, metadata = load_index(index_dir)
    model = SentenceTransformer(MODEL_NAME)

    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, indices = index.search(q_emb, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        record = metadata[idx]
        if role in record.get("allowed_roles", []):
            results.append({**record, "score": float(score)})
        if len(results) >= top_k:
            break

    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--chunks", type=Path, required=True)
    build_p.add_argument("--out", type=Path, required=True)

    query_p = sub.add_parser("query")
    query_p.add_argument("--index", type=Path, required=True)
    query_p.add_argument("--q", type=str, required=True)
    query_p.add_argument("--role", type=str, required=True,
                          help="e.g. engineer, hr, finance, security, manager, support, marketing, sales")
    query_p.add_argument("--top_k", type=int, default=5)

    args = ap.parse_args()

    if args.cmd == "build":
        build_index(args.chunks, args.out)
    elif args.cmd == "query":
        results = search(args.index, args.q, args.role, top_k=args.top_k)
        if not results:
            print(f"No results visible to role='{args.role}' for this query "
                  f"(either no relevant match, or all matches are restricted).")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['score']:.3f}  dept={r['department']}  "
                  f"path={r['source_path']}")
            print(f"    title: {r['title']}")
            print(f"    text:  {r['text'][:200].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()