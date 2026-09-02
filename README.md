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

# 🚀 NexusAI — Enterprise Knowledge & Operations Copilot

![Architecture](https://img.shields.io/badge/Architecture-RAG-blue) ![Deployment](https://img.shields.io/badge/Deployment-HuggingFace_Spaces-orange) ![GPU](https://img.shields.io/badge/Compute-ZeroGPU-green)

NexusAI is a production-grade **Retrieval-Augmented Generation (RAG)** platform built to securely connect enterprise employees with internal documentation, incident reports, and operational knowledge. 

It is engineered from the ground up to handle complex query reasoning, strict role-based access control (RBAC), and high-precision semantic retrieval. The application is fully deployed and accessible on **Hugging Face Spaces**, utilizing the highly scalable **ZeroGPU** infrastructure for real-time model inference.

## 🌟 Key Features

* **Advanced Hybrid Search**: Combines **FAISS dense semantic retrieval** (using `BAAI/bge-small-en-v1.5`) with **BM25 sparse keyword matching**. This ensures highly accurate document retrieval for both conceptual questions and exact-match alphanumeric identifiers (e.g., `INC-1004`, `v2.7.1`).
* **Document-Level Security (ACL)**: Hardened metadata filtering ensures that users can only retrieve and view documents explicitly authorized for their specific role (e.g., `hr`, `engineer`, `manager`). Unauthorized documents are cryptographically invisible to the LLM generation layer.
* **Neural Reranking**: Integrates a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to mathematically rerank the top candidate documents retrieved by the hybrid search engine, guaranteeing the LLM is fed the highest quality contextual signals.
* **Query Rewriting & Intelligence**: Utilizes `meta/llama-3.1-8b-instruct` to asynchronously rewrite ambiguous user queries into highly optimized search vectors, dramatically reducing hallucination rates and improving context precision.
* **Citation Verification Engine**: The system maps LLM-generated claims directly to exact source text chunks, computing a real-time `Citation Validity Rate`. Hallucinated citations are caught and flagged automatically before the user reads them.
* **Automated CI/CD Evaluation**: Leverages LLM-as-a-Judge (measuring Context Precision, Recall, Answer Relevance, and Faithfulness) integrated into a GitHub Actions pipeline to prevent regressions in retrieval quality.

## 🏗️ Architecture & Deployment

The platform is designed to be completely serverless and runs on **Hugging Face Spaces**.

1. **Frontend Layer**: Native asynchronous `gradio.Blocks` interface providing real-time UI streaming, parameter control, and citation rendering.
2. **Compute Layer (ZeroGPU)**: Utilizing Hugging Face's dynamic `spaces` decorator, the heavy embedding and reranking transformers are swapped into active VRAM only when requested, ensuring high-throughput without dedicated GPU overhead.
3. **Database Layer**: Localized FAISS & BM25 `.pkl` indexes hosted directly within the Space environment (`data/index/`). No external vector database connections or persistent networking are required, ensuring zero network latency during the retrieval phase.
4. **Generation Layer**: Python `httpx` async event streams connecting directly to the **Llama 3.1 8B Instruct** model via the blazing-fast Nvidia NIM API.

## 🚀 Live Demo on Hugging Face Spaces

This repository is continuously deployed to Hugging Face Spaces. 

**To experience the Document-Level Security (ACL) in action:**
1. Navigate to the Copilot tab in the UI.
2. Set your **Role (ACL)** to `hr` and ask: *"Give me the root cause of the INC-0001 database outage."* The system will correctly block access to engineering data.
3. Change your **Role (ACL)** to `engineer` and ask the exact same question. The system will retrieve the incident report and generate a grounded, highly-technical summary with accurate citations!

## 🛠️ Local Development & Setup

If you wish to run this pipeline locally:

```bash
# 1. Clone the repository
git clone https://github.com/IyadKouloughli/enterprise-rag-assistant.git
cd enterprise-rag-assistant

# 2. Install minimal dependencies
pip install -r requirements.txt

# 3. Add your LLM API Key
echo "API_KEY=your_nvidia_api_key_here" > .env

# 4. Boot the Gradio application
python run_app.py
```
