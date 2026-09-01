"""
ingest_gitlab_handbook.py

Turns a local clone/zip-extract of gitlab-com/www-gitlab-com into
department-tagged, chunked JSONL records ready for embedding + indexing
in your Enterprise RAG Copilot project.

USAGE
    python ingest_gitlab_handbook.py /path/to/www-gitlab-com --out data/handbook_chunks.jsonl

WHAT IT DOES
    1. Walks the repo looking for markdown handbook content (handles both the
       old Middleman layout `source/handbook/...` and the newer Hugo layout
       `content/handbook/...` — it auto-detects whichever exists).
    2. Skips boilerplate / non-content files (redirects, includes, images).
    3. Strips YAML front matter and Liquid/Hugo template tags, keeping clean
       prose + headings.
    4. Maps each file's path to a `department` bucket using DEPARTMENT_MAP
       below (edit this to fit the taxonomy you want to demo).
    5. Chunks each document (~500 tokens w/ overlap) so chunks are a sane
       size for embeddings.
    6. Writes one JSON object per chunk to a JSONL file with the metadata
       your ACL layer will filter on:
         {
           "document_id": "...",
           "chunk_id": "...",
           "source_path": "...",
           "department": "engineering",
           "allowed_roles": ["engineer", "manager"],
           "title": "...",
           "text": "..."
         }

NEXT STEP AFTER THIS
    Feed handbook_chunks.jsonl into your embedding step (OpenAI/Azure OpenAI
    embeddings or a local sentence-transformers model), then push into your
    vector index (Azure AI Search, FAISS, pgvector, etc.) alongside the
    incident-report and ticket datasets.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Department taxonomy — edit this to match what you want to demo.
#    Key = a substring that appears in the file's path (relative to the
#    handbook root). First match wins. Add/remove rows freely.
# ---------------------------------------------------------------------------
DEPARTMENT_MAP = [
    # path substring                 department        allowed_roles
    ("people-group",                 "hr",             ["hr", "manager"]),
    ("total-rewards",                "hr",             ["hr", "manager"]),
    ("compensation",                 "hr",             ["hr", "manager"]),
    ("hiring",                       "hr",             ["hr", "manager"]),
    ("finance",                      "finance",        ["finance", "manager"]),
    ("legal",                        "legal",          ["legal", "manager"]),
    ("security",                     "security",       ["security", "engineer", "manager"]),
    ("engineering",                  "engineering",     ["engineer", "manager"]),
    ("infrastructure",               "engineering",     ["engineer", "manager"]),
    ("product",                      "engineering",     ["engineer", "manager"]),
    ("support",                      "support",        ["support", "engineer", "manager"]),
    ("marketing",                    "marketing",       ["marketing", "manager"]),
    ("sales",                        "sales",          ["sales", "manager"]),
    ("ceo",                          "exec",           ["manager"]),
    ("leadership",                   "exec",           ["manager"]),
]
DEFAULT_DEPARTMENT = "general"
DEFAULT_ROLES = ["engineer", "manager", "hr", "finance", "security", "support", "marketing", "sales"]

# Files/dirs to ignore outright
SKIP_PATTERNS = [
    r"redirect", r"_index\.md$", r"index\.md$", r"\.svg$", r"\.png$", r"\.jpg$",
]

CHUNK_SIZE_CHARS = 2200     # ~500 tokens
CHUNK_OVERLAP_CHARS = 300


def find_handbook_root(repo_root: Path) -> Path:
    """Auto-detect whether this is the old Middleman layout or new Hugo layout."""
    candidates = [
        repo_root / "source" / "handbook",   # old Middleman layout
        repo_root / "content" / "handbook",  # newer Hugo layout
        repo_root / "handbook",              # standalone handbook repo
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Couldn't find a handbook/ content directory under "
        f"{repo_root}. Run `find {repo_root} -iname '*handbook*' -maxdepth 3` "
        "to locate it manually and pass that path instead."
    )


def should_skip(path: Path) -> bool:
    s = str(path)
    return any(re.search(pat, s, re.IGNORECASE) for pat in SKIP_PATTERNS)


def strip_front_matter(text: str) -> str:
    """Remove YAML front matter (--- ... ---) at the top of the file."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text


def clean_markdown(text: str) -> str:
    text = strip_front_matter(text)
    # Strip Liquid/Hugo template tags like {{ ... }} or {% ... %}
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\{%.*?%\}", "", text)
    # Collapse markdown link syntax [text](url) -> text (keeps prose clean)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_title(text: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


def classify_department(rel_path: str):
    for substr, dept, roles in DEPARTMENT_MAP:
        if substr in rel_path.lower():
            return dept, roles
    return DEFAULT_DEPARTMENT, DEFAULT_ROLES


def chunk_text(text: str, size=CHUNK_SIZE_CHARS, overlap=CHUNK_OVERLAP_CHARS):
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        # try to break on a paragraph boundary near the end
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
    ap.add_argument("repo_root", type=Path, help="Path to extracted www-gitlab-com repo")
    ap.add_argument("--out", type=Path, default=Path("data/handbook_chunks.jsonl"))
    args = ap.parse_args()

    handbook_root = find_handbook_root(args.repo_root)
    print(f"Reading handbook content from: {handbook_root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_chunks = 0
    dept_counts = {}

    with open(args.out, "w", encoding="utf-8") as out_f:
        for md_path in handbook_root.rglob("*.md"):
            if should_skip(md_path):
                continue
            try:
                raw = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                print(f"  skip (read error) {md_path}: {e}")
                continue

            cleaned = clean_markdown(raw)
            if len(cleaned) < 200:
                continue  # too short to be useful content

            rel_path = str(md_path.relative_to(handbook_root))
            department, allowed_roles = classify_department(rel_path)
            title = guess_title(cleaned, fallback=md_path.stem.replace("-", " ").title())
            document_id = doc_id_for(md_path)

            chunks = chunk_text(cleaned)
            for i, chunk in enumerate(chunks):
                record = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{i}",
                    "source_path": rel_path,
                    "department": department,
                    "allowed_roles": allowed_roles,
                    "title": title,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

            dept_counts[department] = dept_counts.get(department, 0) + 1
            n_docs += 1

    print(f"\nDone. {n_docs} documents -> {n_chunks} chunks written to {args.out}")
    print("Documents per department:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"  {dept:12s} {count}")


if __name__ == "__main__":
    main()