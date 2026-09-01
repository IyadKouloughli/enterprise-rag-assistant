"""
ingest_tickets.py (v3 -- fixes the "0 tickets survive" bug from v2)

WHAT WENT WRONG IN v2
    v2 dropped any row containing "{product_purchased}" or "{error_message}".
    Your run showed this dropped ALL 8,469 rows in customer_support_tickets.jsonl
    -- meaning this isn't a defect in a minority of rows, it's a quirk baked
    into the entire file: the dataset's template generator never substituted
    those two fields for ANY ticket. Filtering them out throws away 100% of
    the data.

FIX IN v3
    Instead of dropping rows with placeholders, CLEAN the placeholder text
    out of them (regex substitution) and keep the ticket -- the rest of each
    ticket (subject, status, priority, resolution, etc.) is still real,
    usable content. This gets you actual ticket volume back.

    "Please assist with my {product_purchased}. I'm seeing {error_message}."
    becomes
    "Please assist with my product. I'm seeing an issue."

    Good enough for retrieval purposes -- a RAG demo doesn't need the exact
    product name here, and this is honest about being synthetic data anyway.

STILL FIXED FROM v1
    Explicitly loads only customer_support_tickets.jsonl via data_files=,
    NOT the whole PIISA/dataset repo (which also has an unrelated
    customer_transactions.jsonl file that contaminated v1's output).

SETUP
    pip install datasets

USAGE
    python ingest_tickets.py --out data/ticket_chunks.jsonl
    python ingest_tickets.py --out data/ticket_chunks.jsonl --limit 3000

RE-RUN AFTER THIS
    Get-Content data\\handbook_chunks.jsonl, data\\incident_chunks.jsonl, `
        data\\ticket_chunks.jsonl | Set-Content data\\all_chunks.jsonl
    python build_faiss_index.py build --chunks data\\all_chunks.jsonl --out data\\index
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ALLOWED_ROLES = ["support", "engineer", "manager"]
CHUNK_SIZE_CHARS = 2200
CHUNK_OVERLAP_CHARS = 300
RANDOM_SEED = 42

# Replace unresolved template tokens with natural generic filler instead of
# dropping the whole ticket. Add more pairs here if you spot other
# unresolved {tokens} while spot-checking the output.
PLACEHOLDER_REPLACEMENTS = {
    "{product_purchased}": "the product",
    "{Product Purchased}": "the product",
    "{error_message}": "an error",
    "{Error Message}": "an error",
}


def clean_placeholders(text: str) -> str:
    for token, replacement in PLACEHOLDER_REPLACEMENTS.items():
        text = text.replace(token, replacement)
    # catch any other stray {snake_case_or_words} template tokens generically
    text = re.sub(r"\{[a-zA-Z_ ]+\}", "[unspecified]", text)
    return text


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


def extract_field(text: str, field_name: str):
    m = re.search(rf"{field_name}:\s*(.+)", text)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ticket_chunks.jsonl"))
    ap.add_argument("--limit", type=int, default=5000,
                     help="Max tickets to keep (shuffled sample). Use 0 for no limit.")
    args = ap.parse_args()

    from datasets import load_dataset

    print("Downloading customer_support_tickets.jsonl specifically from PIISA/dataset "
          "(NOT the transactions file)...")
    ds = load_dataset(
        "PIISA/dataset",
        data_files="customer_support_tickets.jsonl",
        split="train",
    )
    print(f"Loaded {len(ds)} raw rows from customer_support_tickets.jsonl")

    rows = [row for row in ds if row.get("text", "").strip()]
    print(f"{len(rows)} rows have non-empty text")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(rows)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]
        print(f"Downsampled to {len(rows)} tickets (--limit {args.limit})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_docs = 0
    n_chunks = 0
    n_cleaned = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for row in rows:
            raw_text = row["text"].strip()
            had_placeholder = "{" in raw_text
            text = clean_placeholders(raw_text)
            if had_placeholder:
                n_cleaned += 1

            ticket_num = extract_field(text, "Ticket ID")
            subject = extract_field(text, "Ticket Subject")
            ticket_id = f"TICKET-{ticket_num}" if ticket_num else f"TICKET-{n_docs}"
            title = f"Ticket {ticket_num}: {subject}" if (ticket_num and subject) else f"Support Ticket {n_docs}"

            document_id = hashlib.sha1(ticket_id.encode("utf-8")).hexdigest()[:12]
            chunks = chunk_text(text)

            for j, chunk in enumerate(chunks):
                record = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_{j}",
                    "ticket_id": ticket_id,
                    "source_path": f"PIISA/dataset/customer_support_tickets.jsonl#{ticket_num or n_docs}",
                    "department": "support",
                    "allowed_roles": ALLOWED_ROLES,
                    "title": title,
                    "text": chunk,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_chunks += 1

            n_docs += 1

    print(f"\nDone. {n_docs} tickets -> {n_chunks} chunks written to {args.out}")
    print(f"({n_cleaned} of {n_docs} had placeholder tokens cleaned up)")


if __name__ == "__main__":
    main()