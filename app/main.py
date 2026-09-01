"""
app/main.py

FastAPI Application for Enterprise AI Knowledge & Operations Copilot:
- REST API endpoints for chat, streaming, document inspection, evaluation benchmark metrics, and healthcheck.
- Serves static interactive UI for browser pair-testing and demos.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentStatsResponse,
    HealthResponse,
)
from generate_answer import build_prompt, call_llm, generate_answer, rewrite_query
from hybrid_search import hybrid_search, load_index_and_metadata
from citation_verifier import CitationVerifier

INDEX_DIR = ROOT_DIR / "data" / "index"
EVAL_RESULTS_PATH = ROOT_DIR / "data" / "eval_results.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="NexusAI — Enterprise Knowledge & Operations Copilot",
    description="Production-grade RAG platform with hybrid search, reranking, ACL document-level security, and citation verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    index_loaded = False
    total_vectors = 0
    try:
        index, meta = load_index_and_metadata(INDEX_DIR)
        index_loaded = True
        total_vectors = index.ntotal
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        model_name=os.environ.get("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct"),
        index_loaded=index_loaded,
        total_indexed_vectors=total_vectors,
    )


def sanitize_numpy(obj: Any) -> Any:
    """Recursively converts NumPy types into Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_numpy(x) for x in obj]
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return obj


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    t0 = time.time()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None

    try:
        result = generate_answer(
            index_dir=INDEX_DIR,
            query=req.query,
            role=req.role,
            history=history_dicts,
            top_k=req.top_k,
            rerank=req.rerank,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - t0, 3)

    return ChatResponse(
        query=result["query"],
        query_used=result["query_used"],
        answer=result["answer"],
        sources=sanitize_numpy(result["sources"]),
        citation_verification=sanitize_numpy(result.get("citation_verification")),
        security_audit=sanitize_numpy(result.get("security_audit")),
        latency_seconds=elapsed,
    )


import httpx


async def stream_llm(system: str, prompt: str, model: str = None, base_url: str = None, api_key: str = None) -> AsyncGenerator[str, None]:
    """Streams token chunks directly from OpenAI-compatible LLM endpoint via non-blocking httpx."""
    api_key = api_key or os.environ.get("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is not set in environment.")

    model = model or os.environ.get("MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")
    base_url = base_url or os.environ.get("BASE_URL", "https://integrate.api.nvidia.com/v1")

    url = f"{base_url.rstrip('/')}/chat/completions"
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


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Real-time streaming endpoint using Server-Sent Events (SSE).
    1. Sends retrieval metadata & audit event
    2. Streams token chunks directly from LLM (Time-to-first-token < 600ms)
    3. Verifies citations and emits completion payload
    """
    t0 = time.time()
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history] if req.history else None

    # Step 1: Optimized Query contextualization & Hybrid retrieval
    effective_query = rewrite_query(req.query, history=history_dicts)
    sources, audit_stats = hybrid_search(
        index_dir=INDEX_DIR,
        query=effective_query,
        role=req.role,
        top_k=req.top_k,
        rerank=req.rerank,
        return_audit=True,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # Emit retrieval & audit event
        retrieval_payload = {
            "type": "retrieval",
            "query_used": effective_query,
            "sources": sanitize_numpy(sources),
            "security_audit": sanitize_numpy(audit_stats),
        }
        yield f"data: {json.dumps(retrieval_payload)}\n\n"

        if not sources:
            roles_str = ", ".join(audit_stats["user_roles"])
            if audit_stats["candidates_blocked_by_acl"] > 0:
                empty_msg = f"Access Restricted: Matching documents exist, but are restricted for role(s) [{roles_str}]."
            else:
                empty_msg = f"No relevant sources found in the knowledge base for role(s) [{roles_str}]."

            yield f"data: {json.dumps({'type': 'token', 'content': empty_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'latency': round(time.time() - t0, 3)})}\n\n"
            return

        # Stream directly from LLM API
        prompt = build_prompt(effective_query, sources)
        accumulated_answer = []

        try:
            async for token in stream_llm(
                system="You are an internal enterprise knowledge assistant. Answer using ONLY provided sources with [1], [2] citations.",
                prompt=prompt,
            ):
                accumulated_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        except Exception as e:
            # Fallback to standard non-streaming call if stream connection breaks
            try:
                full_ans = call_llm(
                    system="You are an internal enterprise knowledge assistant. Answer using ONLY provided sources with [1], [2] citations.",
                    prompt=prompt,
                )
                accumulated_answer = [full_ans]
                yield f"data: {json.dumps({'type': 'token', 'content': full_ans})}\n\n"
            except Exception as ex:
                yield f"data: {json.dumps({'type': 'error', 'message': str(ex)})}\n\n"
                return

        # Verify citation grounding on complete response
        full_text = "".join(accumulated_answer)
        verifier = CitationVerifier()
        verification_report = verifier.verify(full_text, sources)

        done_payload = {
            "type": "done",
            "citation_verification": sanitize_numpy(verification_report),
            "latency": round(time.time() - t0, 3),
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/documents", response_model=DocumentStatsResponse)
def get_documents_stats():
    """Returns department distributions and document counts across the indexed corpus."""
    try:
        _, metadata = load_index_and_metadata(INDEX_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load index: {e}")

    dept_counts: Dict[str, int] = {}
    source_types: Dict[str, int] = {"handbook": 0, "incidents": 0, "tickets": 0}

    for record in metadata:
        dept = record.get("department", "general")
        dept_counts[dept] = dept_counts.get(dept, 0) + 1

        if dept == "incidents" or "incident_id" in record:
            source_types["incidents"] += 1
        elif "ticket_id" in record:
            source_types["tickets"] += 1
        else:
            source_types["handbook"] += 1

    sample_docs = []
    seen_paths = set()
    for record in metadata:
        p = record.get("source_path")
        if p and p not in seen_paths and len(sample_docs) < 15:
            seen_paths.add(p)
            sample_docs.append({
                "title": record.get("title"),
                "department": record.get("department"),
                "source_path": record.get("source_path"),
                "allowed_roles": record.get("allowed_roles"),
            })

    return DocumentStatsResponse(
        total_chunks=len(metadata),
        departments=dept_counts,
        source_types=source_types,
        sample_documents=sample_docs,
    )


@app.get("/api/evaluation")
def get_evaluation_metrics():
    """Returns the latest automated benchmark evaluation metrics and scorecards."""
    if not EVAL_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run eval_rag.py first.")

    with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Serve Static UI files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
