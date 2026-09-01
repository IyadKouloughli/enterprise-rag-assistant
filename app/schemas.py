"""
app/schemas.py

Pydantic models for the Enterprise RAG API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's query string")
    role: str = Field(default="general", description="User role(s) for ACL filtering, e.g. 'hr', 'engineer', 'hr,manager', 'admin'")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Prior conversation messages for multi-turn context")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of source chunks to retrieve")
    rerank: bool = Field(default=True, description="Whether to apply neural CrossEncoder reranking")


class SourceChunk(BaseModel):
    citation_index: Optional[int] = None
    title: str
    department: str
    source_path: str
    text: str
    document_id: Optional[str] = None
    incident_id: Optional[str] = None
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None
    allowed_roles: Optional[List[str]] = None


class CitationVerificationReport(BaseModel):
    status: str
    status_message: str
    total_citation_tags: int
    unique_sources_cited: int
    total_available_sources: int
    citation_validity_rate: float
    sentence_citation_coverage: float
    invalid_citations: List[int] = []


class SecurityAuditReport(BaseModel):
    user_roles: List[str]
    query: str
    total_candidates_evaluated: int
    candidates_passed_acl: int
    candidates_blocked_by_acl: int
    reranking_applied: bool
    top_k_returned: int


class ChatResponse(BaseModel):
    query: str
    query_used: str
    answer: str
    sources: List[Dict[str, Any]]
    citation_verification: Optional[Dict[str, Any]] = None
    security_audit: Optional[Dict[str, Any]] = None
    latency_seconds: float


class DocumentStatsResponse(BaseModel):
    total_chunks: int
    departments: Dict[str, int]
    source_types: Dict[str, int]
    sample_documents: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    index_loaded: bool
    total_indexed_vectors: int
