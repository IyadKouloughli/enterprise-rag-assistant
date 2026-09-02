"""
generate_answer.py

Production-Grade Grounded Answer Generation & Citation Verification:
1. Multi-turn Conversational Query Contextualization / Rewriting.
2. Authorization-Aware Hybrid Retrieval with Cross-Encoder Neural Reranking.
3. Anti-hallucination Prompting for grounded synthesis with inline citations ([1], [2]).
4. Automated Citation Verification & Grounding Telemetry (CitationVerifier).
5. Comprehensive Security & Permission Audit Reporting.

CONFIGURATION (.env)
    API_KEY="your_api_key_here"
    MODEL_NAME="meta/llama-3.2-11b-vision-instruct"
    BASE_URL="https://integrate.api.nvidia.com/v1"

USAGE
    python generate_answer.py --index data/index --q "what is our vacation policy" --role hr
    python generate_answer.py --index data/index --q "INC-0001 database outage" --role engineer --audit
    python generate_answer.py --index data/index --q "vacation rules" --role "hr,manager" --verify-citations
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import re
import httpx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

from dotenv import load_dotenv
load_dotenv()  # reads .env file into os.environ

from citation_verifier import CitationVerifier
from hybrid_search import hybrid_search, parse_user_roles

SYSTEM_INSTRUCTIONS = """You are an internal enterprise knowledge assistant. \
You answer questions using ONLY the numbered sources provided below. \
Follow these rules strictly:

1. Only use information contained in the sources. Do not use outside knowledge.
2. If the sources do not contain enough information to answer, say so explicitly \
instead of guessing or making something up.
3. Cite every claim inline using the source number in square brackets, e.g. "...\
deployed version 2.7.1 [2]." A sentence with no citation should be a connector \
sentence only, not a factual claim.
4. Keep the answer concise and directly address the question -- do not repeat \
the sources verbatim, summarize and synthesize them in your own words.
5. At the end, do not repeat the source list yourself -- it will be appended \
separately.
"""

REWRITE_SYSTEM_PROMPT = """You are a search query rewriting assistant for an enterprise knowledge retrieval engine.
Given the previous conversation history and a new follow-up question, rewrite the question into a self-contained, standalone query for search retrieval.
- Preserve entity names, codes (e.g., INC-0001, v2.7.1), departments, and locations.
- If the question is already fully standalone, output it as-is.
- Output ONLY the rewritten standalone query string with no explanation or preamble.
"""


def build_prompt(query: str, sources: list) -> str:
    source_blocks = []
    for i, s in enumerate(sources, start=1):
        dept = s.get("department", "general")
        path = s.get("source_path", "unknown")
        title = s.get("title", "Untitled")
        text = s.get("text", "")
        source_blocks.append(
            f"[{i}] (department: {dept}, path: {path})\n"
            f"Title: {title}\n"
            f"{text}"
        )
    joined_sources = "\n\n".join(source_blocks)
    return (
        f"SOURCES:\n{joined_sources}\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question using only the sources above, with inline citations."
    )


async def call_llm(
    system: str,
    prompt: str,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    temperature: float = 0.2,
) -> str:
    """Single unified function calling any configured LLM endpoint via async httpx."""
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("Error: API_KEY is not set.")
    api_key = api_key.strip()

    model = model or os.environ.get("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")
    base_url = base_url or os.environ.get("BASE_URL", "https://integrate.api.nvidia.com/v1")
    base_url = base_url.strip().rstrip('/')
    
    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": temperature,
    }

    last_error = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    error_body = resp.text
                    last_error = f"LLM API Error ({resp.status_code}) from {url}: {error_body}"
                    if resp.status_code in [429, 500, 502, 503, 504] and attempt < 3:
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    raise RuntimeError(last_error)
                
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content")
                if not content:
                    content = msg.get("reasoning_content", "")
                return content
        except Exception as e:
            last_error = f"Error calling LLM API ({url}): {e}"
            if attempt < 3:
                import asyncio
                await asyncio.sleep(1.5 * attempt)
                continue
            raise RuntimeError(last_error)

    raise RuntimeError(last_error or "Unknown error calling LLM API")


async def stream_llm(system: str, prompt: str, model: str = None, base_url: str = None, api_key: str = None) -> AsyncGenerator[str, None]:
    """Streams token chunks directly from OpenAI-compatible LLM endpoint via non-blocking httpx."""
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is not set in environment.")
    api_key = api_key.strip()

    model = model or os.environ.get("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")
    base_url = base_url or os.environ.get("BASE_URL", "https://integrate.api.nvidia.com/v1")
    base_url = base_url.strip().rstrip('/')
    
    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                err_bytes = await resp.aread()
                raise RuntimeError(f"LLM API Error ({resp.status_code}): {err_bytes.decode('utf-8', 'replace')}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content") or ""
                        if delta:
                            yield delta
                    except Exception:
                        pass


def needs_contextual_rewriting(query: str, history: Optional[List[Dict[str, str]]]) -> bool:
    """Fast heuristic to determine if query is a dependent follow-up needing LLM rewriting."""
    if not history or len(history) == 0:
        return False

    q_lower = query.lower().strip()
    words = q_lower.split()

    # Short follow-ups (e.g. "what about in Mexico?", "how about expenses?")
    if len(words) <= 5:
        return True

    # Coreference / pronoun indicators
    context_indicators = [
        "it", "its", "they", "them", "their", "this", "that", "these", "those",
        "what about", "how about", "why did that", "where is it", "and for",
        "same for", "instead", "previous", "above"
    ]
    for ind in context_indicators:
        if re.search(r"\b" + re.escape(ind) + r"\b", q_lower):
            return True

    return False


async def rewrite_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> str:
    """Rewrites a follow-up query based on prior conversational turns into a standalone search query."""
    if not needs_contextual_rewriting(query, history):
        return query

    recent_turns = history[-4:]
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_turns])
    prompt = (
        f"CONVERSATION HISTORY:\n{history_str}\n\n"
        f"FOLLOW-UP QUERY: {query}\n\n"
        f"STANDALONE SEARCH QUERY:"
    )

    try:
        rewritten = await call_llm(REWRITE_SYSTEM_PROMPT, prompt, model=model, base_url=base_url, api_key=api_key, temperature=0.1)
        rewritten = rewritten.strip().strip('"').strip("'")
        return rewritten if rewritten else query
    except Exception:
        return query


def generate_answer(
    index_dir: Path,
    query: str,
    role: Union[str, List[str]],
    history: Optional[List[Dict[str, str]]] = None,
    model: str = None,
    base_url: str = None,
    top_k: int = 5,
    rerank: bool = True,
) -> Dict[str, Any]:
    """
    Executes end-to-end grounded generation with security, reranking, and citation verification.

    Returns a rich dictionary containing:
    - query: original user query
    - query_used: standalone rewritten query if history was provided
    - answer: LLM answer text
    - sources: list of retrieved/reranked source chunks
    - citation_verification: verification report from CitationVerifier
    - security_audit: authorization and document filtering audit stats
    """
    # 1. Contextualize query if multi-turn history exists
    effective_query = rewrite_query(query, history=history, model=model, base_url=base_url)

    # 2. Authorization-aware hybrid retrieval + neural reranking
    sources, audit_stats = hybrid_search(
        index_dir=index_dir,
        query=effective_query,
        role=role,
        top_k=top_k,
        rerank=rerank,
        return_audit=True,
    )

    # 3. Handle zero authorized results
    if not sources:
        roles_str = ", ".join(audit_stats["user_roles"])
        if audit_stats["candidates_blocked_by_acl"] > 0:
            msg = (
                f"Access Restricted: Relevant information exists in the knowledge base, "
                f"but is not accessible to your assigned role(s) [{roles_str}]."
            )
        else:
            msg = (
                f"No sources visible to role(s) [{roles_str}] were relevant to this question. "
                f"The requested information was not found in the knowledge base."
            )
        return {
            "query": query,
            "query_used": effective_query,
            "answer": msg,
            "sources": [],
            "citation_verification": {
                "status": "NO_SOURCES",
                "status_message": "No authorized sources retrieved.",
                "total_citation_tags": 0,
                "valid_citations": [],
                "invalid_citations": [],
            },
            "security_audit": audit_stats,
        }

    # 4. Prompt construction & LLM inference
    prompt = build_prompt(effective_query, sources)
    answer = call_llm(SYSTEM_INSTRUCTIONS, prompt, model=model, base_url=base_url)

    # 5. Citation verification & Grounding audit
    verifier = CitationVerifier()
    verification_report = verifier.verify(answer, sources)

    return {
        "query": query,
        "query_used": effective_query,
        "answer": answer,
        "sources": sources,
        "citation_verification": verification_report,
        "security_audit": audit_stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Grounded Answer Generation with Hybrid Search, ACL & Citation Verification")
    ap.add_argument("--index", type=Path, required=True, help="Path to FAISS index directory")
    ap.add_argument("--q", type=str, required=True, help="User query string")
    ap.add_argument("--role", type=str, required=True, help="User role(s), e.g. hr, engineer, or 'hr,manager'")
    ap.add_argument("--model", type=str, default=None, help="Optional model override (defaults to MODEL_NAME in .env)")
    ap.add_argument("--base_url", type=str, default=None, help="Optional base URL override (defaults to BASE_URL in .env)")
    ap.add_argument("--top_k", type=int, default=5, help="Number of retrieved chunks")
    ap.add_argument("--no-rerank", action="store_true", help="Disable neural CrossEncoder reranking")
    ap.add_argument("--verify-citations", action="store_true", default=True, help="Show citation verification breakdown")
    ap.add_argument("--audit", action="store_true", help="Display security and retrieval audit details")
    args = ap.parse_args()

    result = generate_answer(
        index_dir=args.index,
        query=args.q,
        role=args.role,
        model=args.model,
        base_url=args.base_url,
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )

    print("\n" + "=" * 60)
    print("GROUNDED ASSISTANT ANSWER")
    print("=" * 60 + "\n")
    print(result["answer"])

    if result["sources"]:
        print("\n" + "=" * 60)
        print("RETRIEVED & RERANKED SOURCES")
        print("=" * 60)
        for i, s in enumerate(result["sources"], start=1):
            score_info = f"rerank_score={s['rerank_score']:.4f}" if "rerank_score" in s else f"fused_score={s['fused_score']:.4f}"
            print(f"[{i}] {s['title']} | dept: {s.get('department')} | {score_info}")
            print(f"    Path: {s.get('source_path')}")

    if args.verify_citations and result["citation_verification"]:
        cv = result["citation_verification"]
        print("\n" + "-" * 60)
        print(f"CITATION & GROUNDING VERIFICATION: [{cv['status']}]")
        print("-" * 60)
        print(f"  Status: {cv['status_message']}")
        print(f"  Total Citations: {cv.get('total_citation_tags', 0)} | Unique Sources Cited: {cv.get('unique_sources_cited', 0)}")
        print(f"  Sentence Citation Coverage: {cv.get('sentence_citation_coverage', 0) * 100:.1f}%")
        if cv.get("invalid_citations"):
            print(f"  ⚠️ Warning: Invalid Citation IDs Detected: {cv['invalid_citations']}")

    if args.audit and result["security_audit"]:
        sa = result["security_audit"]
        print("\n" + "-" * 60)
        print("SECURITY & ACL AUDIT")
        print("-" * 60)
        print(f"  User Roles: {sa['user_roles']}")
        print(f"  Total Candidates Evaluated: {sa['total_candidates_evaluated']}")
        print(f"  Passed ACL: {sa['candidates_passed_acl']}")
        print(f"  Blocked by ACL: {sa['candidates_blocked_by_acl']}")
        print(f"  Neural Reranking: {sa['reranking_applied']}")


if __name__ == "__main__":
    main()