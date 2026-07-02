# Approach Document: SHL Assessment Recommender Agent

## 1. Executive Summary

This document outlines the architecture and implementation of an autonomous, context-aware Retrieval-Augmented Generation (RAG) agent designed to recommend SHL assessments through dynamic dialogue. The system leverages a local vector database for fast semantic retrieval and an ultra-low latency inference engine to handle multi-turn conversations, filtering, and guardrail enforcement while strictly adhering to a machine-readable JSON schema.

---

## 2. System Architecture

The application is built on a decoupled, stateless architecture comprising four core components:

```
[User Request] ──> [FastAPI Server (main.py)] 
                          │
                          ▼
            [Reasoning Engine (agent.py)] 
             ├──> [Semantic Search (FAISS Index)]
             └──> [Inference Pipeline (Groq Cloud API)]

```

* **API Layer (`main.py`):** A stateless FastAPI implementation exposing `/health` and `/chat` endpoints, utilizing Pydantic models for strict type validation and request/response schema compliance.
* **Vector Database (`indexer.py`):** A lightweight, high-performance dense vector index using Facebook AI Similarity Search (FAISS).
* **Reasoning Engine (`agent.py`):** Controls the contextual assembly, state management via conversational history passage, and inference routing.

---

## 3. Detailed Component Design

### A. Data Ingestion & Normalization

* **Source Data:** Processed from the raw JSON payload endpoint, bypassing standard HTML scraping pipelines to prevent fragile DOM-dependency issues and circumvent CDN/WAF blocks (e.g., CloudFront).
* **Parsing Strategy:** Implemented robust streaming text parsing with non-strict control-character handling (`strict=False`) to normalize hidden formatting anomalies within text fields.

### B. Embedding Strategy & Indexing

* **Model Selection:** Utilized `sentence-transformers/all-MiniLM-L6-v2` to map text attributes into a dense 384-dimensional vector space. This choice ensures an optimal balance between embedding quality and local CPU performance.
* **Document Schema Configuration:** To maximize keyword alignment and semantic density during retrieval, metadata attributes were unified into a structured text document per item:

$$\text{Search Text} = \text{"Test Name: \{name\}. Type: \{keys\}. Job Levels: \{levels\}. Description: \{desc\}"}$$


* **Index Architecture:** Built using an Exact Search L2 distance metric (`faiss.IndexFlatL2`), guaranteeing precise distance calculations for small-to-medium datasets (~377 records) without the quantization loss associated with approximate nearest neighbor (ANN) indexes.

### C. Contextual Retrieval & Dialogue Routing

* **Conversational Memory Loop:** The stateless API acts as a pass-through for the full dialog array. On every turn, the latest user utterance is isolated to query the FAISS database, while the entire history is preserved and fed to the LLM to maintain continuity.
* **Structured Output Generation:** Utilizing Groq's specialized structured JSON mode paired with custom-tailored system instructions. This ensures that the generated output natively adheres to the mandatory layout:

```json
{
    "reply": "string",
    "recommendations": [{"name": "string", "url": "string", "test_type": "string"}],
    "end_of_conversation": "boolean"
}

```

---

## 4. Engineering Tradeoffs & Optimizations

### A. Latency vs. Model Scale

* **Decision:** Selected optimized open models (e.g., `openai/gpt-oss-20b`) via Groq's inference engine over heavier foundational models ($70\text{B}+$ parameters).
* **Justification:** The core requirement of this application is routing, schema mapping, and contextual reasoning rather than open-ended text synthesis. Utilizing a high-throughput, medium-scale model keeps response times well under the evaluation deadline while maintaining deterministic JSON parsing capabilities.

### B. Stateful Behavior in a Stateless Endpoint

* **Decision:** Shifted memory overhead entirely to the client-side request payload.
* **Justification:** Eliminates the need for server-side session tracking databases (like Redis), creating a highly scalable, horizontally deployable container instance.

---

## 5. Security & Safety Guardrails

* **Regulatory Refusal:** Explicit system prompt instructions prevent the agent from offering legal counsel or compliance vetting (e.g., interpreting HIPAA mandates).
* **Prompt Injection Mitigation:** System configurations isolate the dynamic RAG context block away from user message blocks, preventing execution overrides from malicious user prompts. 