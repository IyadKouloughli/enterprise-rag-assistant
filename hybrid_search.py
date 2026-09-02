"""
hybrid_search.py

Production-Grade Authorization-Aware Hybrid Search & Semantic Reranking:
1. Dense Vector Search (FAISS + BAAI/bge-small-en-v1.5) for semantic similarity.
2. Sparse Lexical Search (BM25Okapi) for exact token matching (e.g. INC-1842, v2.7.1).
3. Reciprocal Rank Fusion (RRF) to merge heterogeneous vector & lexical ranking spaces.
4. Document-Level Security (ACL Filtering): Multi-role verification & admin wildcard.
5. Neural Cross-Encoder Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2) for top precision.
6. Security Audit & Observability Telemetry (tracking ACL-blocked documents).

USAGE
    python hybrid_search.py --index data/index --q "vacation policy" --role hr
    python hybrid_search.py --index data/index --q "INC-1842" --role engineer --rerank
    python hybrid_search.py --index data/index --q "salary and compensation" --role "hr,manager"
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import spaces

# Global cache for heavy models and indexes to avoid reloading in persistent processes
_FAISS_INDEX_CACHE = {}
_METADATA_CACHE = {}
_BM25_CACHE = {}
_EMBED_MODEL_CACHE = None
_RERANKER_CACHE = None


def get_embedding_model():
    global _EMBED_MODEL_CACHE
    if _EMBED_MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL_CACHE = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL_CACHE


def get_reranker_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _RERANKER_CACHE
    if _RERANKER_CACHE is None:
        from sentence_transformers import CrossEncoder
        _RERANKER_CACHE = CrossEncoder(model_name)
    return _RERANKER_CACHE


def load_index_and_metadata(index_dir: Path):
    index_key = str(index_dir.resolve())
    if index_key in _FAISS_INDEX_CACHE and index_key in _METADATA_CACHE:
        return _FAISS_INDEX_CACHE[index_key], _METADATA_CACHE[index_key]

    import faiss

    faiss_path = index_dir / "embeddings.faiss"
    meta_path = index_dir / "metadata.jsonl"

    if not faiss_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing index files in {index_dir}. Expected embeddings.faiss and metadata.jsonl.")

    index = faiss.read_index(str(faiss_path))
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata.append(json.loads(line))

    _FAISS_INDEX_CACHE[index_key] = index
    _METADATA_CACHE[index_key] = metadata
    return index, metadata


def build_bm25(metadata, index_dir: Path = None, index_key: str = "default"):
    if index_key in _BM25_CACHE:
        return _BM25_CACHE[index_key]

    import pickle

    # Check for pre-cached BM25 on disk
    if index_dir:
        bm25_cache_file = index_dir / "bm25.pkl"
        if bm25_cache_file.exists():
            try:
                with open(bm25_cache_file, "rb") as f:
                    bm25 = pickle.load(f)
                _BM25_CACHE[index_key] = bm25
                return bm25
            except Exception:
                pass

    from rank_bm25 import BM25Okapi

    corpus_tokens = [m["text"].lower().split() for m in metadata]
    bm25 = BM25Okapi(corpus_tokens)
    _BM25_CACHE[index_key] = bm25

    # Persist to disk for instant future loads
    if index_dir:
        try:
            with open(index_dir / "bm25.pkl", "wb") as f:
                pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return bm25


def parse_user_roles(role_input: Union[str, List[str], Set[str]]) -> Set[str]:
    """Parses single role, comma-separated roles, or list into a clean set of lowercase roles."""
    if isinstance(role_input, (list, set, tuple)):
        roles = {str(r).strip().lower() for r in role_input if str(r).strip()}
    elif isinstance(role_input, str):
        roles = {r.strip().lower() for r in role_input.split(",") if r.strip()}
    else:
        roles = set()
    return roles


def check_acl_permission(user_roles: Set[str], allowed_roles: List[str]) -> bool:
    """
    Checks if any user role satisfies the document's allowed_roles.
    Supports 'admin' / '*' wildcard authorization for superusers.
    """
    if "admin" in user_roles or "*" in user_roles or "all" in user_roles:
        return True

    allowed_set = {str(r).strip().lower() for r in allowed_roles}
    # If the document has no role restrictions, it's public to all
    if not allowed_set or "all" in allowed_set or "*" in allowed_set:
        return True

    return bool(user_roles.intersection(allowed_set))


def reciprocal_rank_fusion(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    """
    Reciprocal Rank Fusion (RRF) combines ranked lists without needing normalized scores:
        RRF_Score(d) = sum(1 / (k + rank(d)))
    """
    scores = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Dict[str, Any]]:
    """
    Reranks candidate chunks using a neural CrossEncoder model.
    Evaluates full attention across (query, text) pairs.
    """
    if not candidates:
        return []

    if len(candidates) <= 1:
        return candidates

    reranker = get_reranker_model(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])

    reranked = sorted(candidates, key=lambda x: -x["rerank_score"])
    return reranked[:top_k]


@spaces.GPU
def hybrid_search(
    index_dir: Path,
    query: str,
    role: Union[str, List[str]],
    top_k: int = 5,
    fetch_k: int = 50,
    rerank: bool = True,
    return_audit: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Executes authorization-aware hybrid search with neural reranking.

    Steps:
    1. Dense Vector Search + Sparse BM25 Search (top fetch_k each)
    2. Reciprocal Rank Fusion (RRF)
    3. Document-Level ACL Security Filtering
    4. Cross-Encoder Neural Reranking (optional, on top authorized candidates)
    5. Returns Top-K results and security audit telemetry.
    """
    user_roles = parse_user_roles(role)
    index_path = Path(index_dir)
    index_key = str(index_path.resolve())
    index, metadata = load_index_and_metadata(index_path)
    bm25 = build_bm25(metadata, index_dir=index_path, index_key=index_key)

    # --- 1. Dense Vector Search ---
    embed_model = get_embedding_model()
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    _, vec_indices = index.search(q_emb, min(fetch_k, index.ntotal))
    vec_ranked = [int(i) for i in vec_indices[0] if i >= 0]

    # --- 2. Sparse BM25 Search ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = list(np.argsort(bm25_scores)[::-1][:fetch_k])

    # --- 3. Reciprocal Rank Fusion ---
    fused_scores = reciprocal_rank_fusion([vec_ranked, bm25_ranked])
    fused_sorted = sorted(fused_scores.items(), key=lambda x: -x[1])

    # --- 4. ACL Security Filtering ---
    authorized_candidates = []
    blocked_count = 0
    total_evaluated = 0

    # Optimized reranker candidate pool (max 10 to minimize CPU compute lag)
    candidate_pool_limit = min(max(top_k * 2, 8), 10) if rerank else top_k

    for idx, score in fused_sorted:
        total_evaluated += 1
        record = metadata[idx]
        is_authorized = check_acl_permission(user_roles, record.get("allowed_roles", []))

        if is_authorized:
            authorized_candidates.append({**record, "fused_score": float(score)})
            if len(authorized_candidates) >= candidate_pool_limit:
                break
        else:
            blocked_count += 1

    # --- 5. Cross-Encoder Reranking ---
    if rerank and len(authorized_candidates) > 0:
        final_results = rerank_candidates(query, authorized_candidates, top_k=top_k)
    else:
        final_results = authorized_candidates[:top_k]

    audit_stats = {
        "user_roles": sorted(list(user_roles)),
        "query": query,
        "total_candidates_evaluated": total_evaluated,
        "candidates_passed_acl": len(authorized_candidates),
        "candidates_blocked_by_acl": blocked_count,
        "reranking_applied": rerank,
        "top_k_returned": len(final_results),
    }

    if return_audit:
        return final_results, audit_stats
    return final_results


def main():
    ap = argparse.ArgumentParser(description="Authorization-Aware Hybrid Search + Reranking")
    ap.add_argument("--index", type=Path, required=True, help="Path to index directory")
    ap.add_argument("--q", type=str, required=True, help="Search query")
    ap.add_argument("--role", type=str, required=True, help="User role(s), e.g. hr, engineer, or 'hr,engineer'")
    ap.add_argument("--top_k", type=int, default=5, help="Number of final results to return")
    ap.add_argument("--no-rerank", action="store_true", help="Disable CrossEncoder neural reranking")
    ap.add_argument("--audit", action="store_true", help="Display security and retrieval audit telemetry")
    args = ap.parse_args()

    results, audit = hybrid_search(
        args.index,
        args.q,
        args.role,
        top_k=args.top_k,
        rerank=not args.no_rerank,
        return_audit=True,
    )

    print("\n" + "=" * 60)
    print(f"SEARCH RESULTS FOR ROLE(S): {audit['user_roles']}")
    print("=" * 60)

    if not results:
        print(f"\nNo results visible to role(s) '{args.role}' for this query.")
        if audit["candidates_blocked_by_acl"] > 0:
            print(f"Notice: {audit['candidates_blocked_by_acl']} matching document(s) were blocked by ACL policy.")

    for i, r in enumerate(results, 1):
        score_info = f"rerank_score={r['rerank_score']:.4f}" if "rerank_score" in r else f"fused_score={r['fused_score']:.4f}"
        print(f"\n[{i}] {score_info} | dept={r['department']} | path={r['source_path']}")
        print(f"    Title: {r['title']}")
        print(f"    Allowed Roles: {r.get('allowed_roles')}")
        print(f"    Excerpt: {r['text'][:220].replace(chr(10), ' ')}...")

    if args.audit:
        print("\n" + "-" * 60)
        print("SECURITY & RETRIEVAL AUDIT")
        print("-" * 60)
        for k, v in audit.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
