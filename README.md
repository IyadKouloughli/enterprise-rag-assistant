# NexusAI — Enterprise Knowledge & Operations Copilot

NexusAI is a production-grade **Retrieval-Augmented Generation (RAG)** platform built to securely connect enterprise employees with internal documentation, incident reports, and operational knowledge. 

It is engineered from the ground up to handle complex query reasoning, strict role-based access control (RBAC), and high-precision semantic retrieval. The application is deployed and accessible on **Hugging Face Spaces**, utilizing **ZeroGPU** infrastructure for real-time model inference.

## Key Features

* **Advanced Hybrid Search**: Combines **FAISS dense semantic retrieval** (using `BAAI/bge-small-en-v1.5`) with **BM25 sparse keyword matching**. This ensures highly accurate document retrieval for both conceptual questions and exact-match alphanumeric identifiers (e.g., `INC-1004`, `v2.7.1`).
* **Document-Level Security (ACL)**: Hardened metadata filtering ensures that users can only retrieve and view documents explicitly authorized for their specific role (e.g., `hr`, `engineer`, `manager`). Unauthorized documents are invisible to the LLM generation layer.
* **Neural Reranking**: Integrates a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to mathematically rerank the top candidate documents retrieved by the hybrid search engine, guaranteeing the LLM is fed the highest quality contextual signals.
* **Query Rewriting & Intelligence**: Utilizes `meta/llama-3.1-8b-instruct` to asynchronously rewrite ambiguous user queries into highly optimized search vectors, dramatically reducing hallucination rates and improving context precision.
* **Citation Verification Engine**: Maps LLM-generated claims directly to exact source text chunks, computing a real-time `Citation Validity Rate`. Hallucinated citations are detected and flagged automatically before the user reads them.
* **Automated CI/CD Evaluation**: Leverages LLM-as-a-Judge (measuring Context Precision, Recall, Answer Relevance, and Faithfulness) integrated into an evaluation suite to prevent regressions in retrieval quality.

## Architecture & Deployment

The platform is designed to be completely serverless and runs on **Hugging Face Spaces**.

1. **Frontend Layer**: Native asynchronous `gradio.Blocks` interface providing real-time UI streaming, parameter control, and citation rendering.
2. **Compute Layer (ZeroGPU)**: Utilizing Hugging Face's dynamic `spaces` decorator, the heavy embedding and reranking transformers are swapped into active VRAM only when requested, ensuring high throughput without dedicated GPU overhead.
3. **Database Layer**: Localized FAISS & BM25 `.pkl` indexes hosted directly within the application environment (`data/index/`). No external vector database connections or persistent networking are required, ensuring zero network latency during retrieval.
4. **Generation Layer**: Python `httpx` async event streams connecting directly to the **Llama 3.1 8B / Llama 3.2 11B** models via OpenAI-compatible API endpoints (NVIDIA NIM, OpenAI, or local Ollama).

## Testing on Hugging Face Spaces (No Installation Required)

The fastest way to test the platform is through the live deployment on Hugging Face Spaces. No local environment setup, Git cloning, or API key configuration is required:

**Live Application URL**: [https://huggingface.co/spaces/iyadkl/enterprise-copilot](https://huggingface.co/spaces/iyadkl/enterprise-copilot)

### Verification Examples to Try in the Web UI:

1. **Test Role-Based Access Control (RBAC)**:
   * In the left panel, set **Your Role (ACL)** to `hr` and submit:
     > *"What caused the Travis CI production database truncation incident?"*
     * **Expected Result**: Access is blocked by ACL filtering (`hr` role has no access to engineering postmortems).
   * Change **Your Role (ACL)** to `engineer` and ask the exact same question:
     * **Expected Result**: Access is granted. The system retrieves the incident report and streams the technical root cause with inline citation `[1]`.

2. **Test Policy & Operations Retrieval**:
   * Set **Your Role (ACL)** to `hr` or `manager` and submit:
     > *"What is our policy and approval workflow for expense reimbursements?"*
     * **Expected Result**: Retrieves internal policy documentation with exact source references.

3. **Test Exact Alphanumeric Identifiers (Hybrid Search)**:
   * Set **Your Role (ACL)** to `engineer` and submit:
     > *"Summarize postmortem incident INC-1004."*
     * **Expected Result**: BM25 keyword matching directly targets the specific incident code, while FAISS retrieves related operational context.

4. **Inspect Knowledge Base Distribution**:
   * Navigate to the **Document Statistics** tab and click **Refresh Statistics** to view the real-time distribution of indexed documents across departments.

## Local Development & Setup

If you wish to run the application or evaluation suite locally on your machine:

### 1. Prerequisites
* Python 3.10 or 3.11
* Git LFS (required to pull the binary FAISS and BM25 index files)

### 2. Clone Repository & Pull Index Data
```bash
git lfs install
git clone https://github.com/IyadKouloughli/enterprise-rag-assistant.git
cd enterprise-rag-assistant
git lfs pull
```

### 3. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file by copying the provided example:
```bash
cp .env.example .env
```
Set your credentials in `.env`:
```env
API_KEY=your_nvidia_api_key_here
MODEL_NAME=meta/llama-3.2-11b-vision-instruct
BASE_URL=https://integrate.api.nvidia.com/v1
```

### 5. Run the Application
Start the local Gradio interface:
```bash
python run_app.py
```
Open `http://127.0.0.1:7860` in your browser.

### 6. Local CLI Verification Examples

* **Test Hybrid Retrieval & Access Control (No API Key Required)**:
  ```bash
  # As HR (restricted access):
  python hybrid_search.py --index data/index --q "Travis CI production database truncation" --role hr

  # As Engineer (authorized access with neural reranking):
  python hybrid_search.py --index data/index --q "Travis CI production database truncation" --role engineer --rerank
  ```

* **Test Grounded Answer Generation & Citation Verifier**:
  ```bash
  python generate_answer.py --index data/index --q "What caused the Travis CI production database truncation incident?" --role engineer --audit
  ```

* **Run Automated Evaluation Benchmark**:
  ```bash
  python eval_rag.py --benchmark data/eval_benchmark.json --index data/index --top_k 5 --retrieval-only
  ```
