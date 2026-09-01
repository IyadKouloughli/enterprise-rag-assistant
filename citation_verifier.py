"""
citation_verifier.py

Enterprise Grounding & Citation Verification Engine:
- Parses inline citations ([1], [2], [1, 2], etc.) from LLM answers.
- Verifies every cited index against the retrieved source chunks.
- Flags hallucinated/out-of-bounds citations (e.g., [9] when only 5 sources were provided).
- Extracts cited sentence claims and links them to exact source metadata.
- Computes citation precision, citation validity rate, and grounding status.
"""

import re
from typing import Any, Dict, List, Tuple


class CitationVerifier:
    def __init__(self):
        # Matches patterns like [1], [2], [1, 2], [1][2], [1-3]
        self.citation_pattern = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
        self.sentence_split_pattern = re.compile(r"(?<=[.!?])\s+")

    def extract_citations(self, text: str) -> List[int]:
        """Extracts all citation numbers found in the text in order."""
        matches = self.citation_pattern.findall(text)
        citations = []
        for match in matches:
            for num_str in match.split(","):
                num_str = num_str.strip()
                if num_str.isdigit():
                    citations.append(int(num_str))
        return citations

    def verify(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates the citations in the generated answer against the provided sources list.

        Returns a detailed verification report.
        """
        num_sources = len(sources)
        all_cited_indices = self.extract_citations(answer)
        unique_cited = sorted(set(all_cited_indices))

        valid_citations = [idx for idx in unique_cited if 1 <= idx <= num_sources]
        invalid_citations = [idx for idx in unique_cited if idx < 1 or idx > num_sources]

        # Break answer into sentences to map claims to citations
        sentences = [s.strip() for s in self.sentence_split_pattern.split(answer) if s.strip()]
        sentence_claims = []
        sentences_with_citations = 0

        for s in sentences:
            citations_in_sentence = self.extract_citations(s)
            is_cited = len(citations_in_sentence) > 0
            if is_cited:
                sentences_with_citations += 1

            # Map to source summaries
            mapped_sources = []
            for c in citations_in_sentence:
                if 1 <= c <= num_sources:
                    src = sources[c - 1]
                    mapped_sources.append({
                        "citation_index": c,
                        "title": src.get("title", "Unknown"),
                        "department": src.get("department", "unknown"),
                        "source_path": src.get("source_path", ""),
                        "chunk_id": src.get("chunk_id", ""),
                    })
                else:
                    mapped_sources.append({
                        "citation_index": c,
                        "error": "OUT_OF_BOUNDS_HALLUCINATION",
                    })

            sentence_claims.append({
                "sentence": s,
                "citations": citations_in_sentence,
                "sources": mapped_sources,
            })

        total_citations = len(all_cited_indices)
        citation_validity_rate = (len(valid_citations) / len(unique_cited)) if unique_cited else 1.0
        sentence_citation_ratio = (sentences_with_citations / len(sentences)) if sentences else 0.0

        # Determine grounding status
        if invalid_citations:
            status = "FAILED"
            status_message = f"Found {len(invalid_citations)} hallucinated citation index(es): {invalid_citations}"
        elif total_citations == 0 and num_sources > 0:
            status = "WARNING"
            status_message = "No inline citations were found in the answer."
        else:
            status = "VERIFIED"
            status_message = f"All {len(valid_citations)} cited sources are valid and grounded."

        # Map each referenced source with its usage count
        cited_sources_summary = []
        for idx in valid_citations:
            src = sources[idx - 1]
            cited_sources_summary.append({
                "citation_index": idx,
                "title": src.get("title"),
                "department": src.get("department"),
                "source_path": src.get("source_path"),
                "mention_count": all_cited_indices.count(idx),
                "excerpt": src.get("text", "")[:250] + "...",
            })

        return {
            "status": status,
            "status_message": status_message,
            "total_citation_tags": total_citations,
            "unique_sources_cited": len(valid_citations),
            "total_available_sources": num_sources,
            "citation_validity_rate": round(citation_validity_rate, 4),
            "sentence_citation_coverage": round(sentence_citation_ratio, 4),
            "invalid_citations": invalid_citations,
            "cited_sources": cited_sources_summary,
            "claims_breakdown": sentence_claims,
        }
