---
title: NexusAI Copilot
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.11"
app_file: run_app.py
pinned: false
---

# NexusAI — Enterprise Knowledge & Operations Copilot

NexusAI is a production-grade Retrieval-Augmented Generation (RAG) platform built to securely connect enterprise employees with internal documentation, incident reports, and operational knowledge. 

It is designed to run seamlessly on Hugging Face Spaces (Gradio + ZeroGPU) using a localized vector database.

## 🌟 Key Features

* **Advanced Hybrid Search**: Combines FAISS dense semantic retrieval with BM25 sparse keyword matching, ensuring highly accurate document retrieval for both conceptual questions and exact-match identifiers (e.g., `INC-1004`).
* **Document-Level Security (ACL)**: Hardened metadata filtering ensures that users can only retrieve and view documents authorized for their specific role (e.g., `hr`, `engineer`, `manager`). Unauthorized documents are completely invisible to the LLM.
* **Neural Reranking**: Integrates an MS-MARCO Cross-Encoder to mathematically rerank the top candidate documents retrieved by the hybrid search, guaranteeing the LLM is fed the highest quality context.
* **Citation Verification**: The engine maps LLM-generated claims to exact source text chunks and computes a real-time `Citation Validity Rate`. Hallucinated citations are caught and flagged automatically.
* **Automated CI/CD Evaluation**: Uses LLM-as-a-Judge (Context Precision, Recall, Answer Relevance, and Faithfulness) integrated into a GitHub Actions pipeline (`eval_rag.py`) to prevent regression in retrieval quality.

## 🏗️ Architecture

1. **Frontend**: Native `gradio.Blocks` interface providing real-time LLM token streaming and citation verification.
2. **Backend**: Python `httpx` async streams connecting directly to Nvidia NIM (Llama 3.2 11B Vision).
3. **Database**: Local FAISS & BM25 `.pkl` indexes hosted directly within the Space environment (`data/index/`). No external database connections required.

## 🚀 Running on Hugging Face Spaces

This repository is optimized for deployment on Hugging Face Spaces using the **ZeroGPU** hardware tier.

### Setup Instructions
1. Create a new Hugging Face Space.
2. Set the SDK to **Gradio**.
3. Under the **Settings > Variables and secrets** tab, add your LLM API Key:
   - `API_KEY`: Your Nvidia API Key (`nvapi-...`)
4. **Push** this repository to the Space. 
5. The `spaces` library will automatically request ZeroGPU access for the FAISS embedding and Cross-Encoder models during runtime.

*(Note: Ensure you do NOT set `BASE_URL` or `MODEL_NAME` secrets unless you are explicitly overriding the defaults. The app automatically configures the correct endpoint for `meta/llama-3.2-11b-vision-instruct`)*.

## 🧪 Testing the Copilot
Once the Space is running, test the ACL security by switching roles in the interface:
- **Role `hr`**: Ask for details on an engineering database outage. The system will block the request.
- **Role `engineer`**: Ask the same question. The system will retrieve the incident report and generate a grounded summary with citations!
