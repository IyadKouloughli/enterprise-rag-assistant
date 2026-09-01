"""
ingest_incidents.py

Converts a local clone of icco/postmortems (a maintained, structured
expansion of the well-known danluu/post-mortems collection of real public
incident postmortems) into department-tagged, chunked JSONL records using
the SAME schema as ingest_gitlab_handbook.py, so you can concatenate them
into one file before embedding.

SETUP
    git clone https://github.com/icco/postmortems.git

USAGE
    python ingest_incidents.py ./postmortems --out data/incident_chunks.jsonl

WHAT IT DOES
    - Walks the repo's `data/` directory (markdown postmortem files).
    - Strips YAML front matter but keeps its fields (company, date, tags)
      as part of the record for context.
    - Assigns a sequential INC-XXXX id per file (in filename order) so you
      get the same "look up INC-1842" experience as the original project
      plan described, plus the source_path for traceability.
    - Tags every incident record with department="incidents" and
      allowed_roles covering the technical/ops roles (engineer, manager,
      security, support) -- edit ROLES_FOR_INCIDENTS below if you want a
      narrower demo (e.g. restrict some to security-only).
    - Chunks long postmortems the same way the handbook script does.

OUTPUT SCHEMA (identical fields to handbook_chunks.jsonl, plus incident_id)
    {
      "document_id": "...",
      "chunk_id": "...",
      "incident_id": "INC-0001",
      "source_path": "...",
      "department": "incidents",
      "allowed_roles": [...],
      "title": "...",
      "text": "..."
    }

NEXT STEP
    Concatenate with the handbook + ticket chunk files before building the
    FAISS index, e.g. on Windows PowerShell:
        Get-Content data\\handbook_chunks.jsonl, data\\incident_chunks.jsonl, `
            data\\ticket_chunks.jsonl | Set-Content data\\all_chunks.jsonl
    Then: python build_faiss_index.py build --chunks data\\all_chunks.jsonl --out data\\index
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROLES_FOR_INCIDENTS = ["engineer", "manager", "security", "support"]
CHUNK_SIZE_CHARS = 2200
CHUNK_OVERLAP_CHARS = 300


def strip_front_matter(text: str):
    """Returns (front_matter_dict, remaining_text). Very small YAML subset
    parser -- good enough for simple key: value front matter blocks."""
    front_matter = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_fm, body = parts[1], parts[2]
            for line in raw_fm.strip().splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    front_matter[key.strip()] = val.strip().strip('"').strip("'")
            return front_matter, body.strip()
    return front_matter, text


def clean_markdown(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_title(text: str, front_matter: dict, fallback: str) -> str:
    for key in ("title", "Title", "company", "Company"):
        if key in front_matter and front_matter[key]:
            return front_matter[key]
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def chunk_text(text: str, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        last_break = chunk.rfind("\n\n")
        if last_break > size * 0.5:
            chunk = chunk[:last_break]
            end = start + last_break
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def doc_id_for(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", type=Path, help="Path to cloned postmortems repo")
    ap.add_argument("--out", type=Path, default=Path("data/incident_chunks.jsonl"))
    args = ap.parse_args()

    data_dir = args.repo_root / "data"
    if not data_dir.exists():
        # fall back to scanning the whole repo for markdown if layout differs
        data_dir = args.repo_root
        print(f"No 'data' subfolder found -- scanning entire {args.repo_root} instead")

    md_files = sorted(data_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files under {data_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_chunks = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, md_path in enumerate(md_files, start=1):
            try:
                raw = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"  skip (read error) {md_path}: {e}")
                continue

            front_matter, body = strip_front_matter(raw)
            cleaned = clean_markdown(body)
            if len(cleaned) < 150:
                continue

            incident_id = f"INC-{i:04d}"
            title = guess_title(cleaned, front_matter, fallback=md_path.stem.replace("-", " ").title())
            document_id = doc_id_for(md_path)
            rel_path = str(md_path.relative_to(args.repo_root))

            chunks = chunk_text(cleaned)
            for j, chunk in enumerate(chunks):
                record = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{j}",
                    "incident_id": incident_id,
                    "source_path": rel_path,
                    "department": "incidents",
                    "allowed_roles": ROLES_FOR_INCIDENTS,
                    "title": title,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_docs += 1

    print(f"\nDone. {n_docs} incidents -> {n_chunks} chunks written to {args.out}")


if __name__ == "__main__":
    main()
