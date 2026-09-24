# AI Support Knowledge Base Service — Document Specification Sheet

**Version:** 1.0.0  
**Target Environment:** Python 3.10+ / Windows / Linux / macOS  
**Primary Components:** FAISS Vector Index, OpenAI Embeddings (`text-embedding-3-small`), Grounded LLM Generation, Deterministic Guardrails & Citation Validator.

---

## 1. System Overview & Architecture

The service is a production-minded, retrieval-augmented question answering (RAG) system built for an internal support desk. It ingests local product policy documentation, indexes content chunks using dense vector embeddings in FAISS, retrieves top relevant passages with cosine similarity scores, constructs a strictly grounded prompt, generates concise answers with citations, and deterministically validates citations and refusal compliance.

```
+-----------------------------------------------------------------------------------+
|                              1. Ingestion Pipeline                                |
|  docs/*.md  -->  Document Loader  -->  Sliding-Window Chunker  -->  Metadata Tag  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        2. Vector Database & Indexing                              |
|  OpenAI Embedding (text-embedding-3-small) --> FAISS IndexFlatIP (Cosine Metric)  |
+-----------------------------------------------------------------------------------+
                                         |
                     [ User Question / Evaluation Suite ]
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              3. Retrieval Stage                                   |
|  Query Embedding --> Top-K Similarity Search --> Score Filtering & Deduplication  |
|                     Output: retrieval_results.json                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                      4. Grounded Answer Generation                                |
|  Context-Bound Prompt Template --> LLM (Strict No-Hallucination Policy)          |
|                     Output: answers.json                                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     5. Deterministic Validation Guardrail                         |
|  - Non-empty answer check                                                         |
|  - Cited document membership in retrieved set                                     |
|  - Mandatory citation for supported claims                                        |
|  - Unsupported / refusal consistency verification                                 |
|                     Output: validation_report.json                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              6. Interfaces                                        |
|  - CLI Interface: python app.py --question "..."                                  |
|  - REST API: FastAPI /ask endpoint                                                |
+-----------------------------------------------------------------------------------+
```

---

## 2. Document Corpus Specification (`docs/`)

The corpus consists of 5 realistic, domain-specific platform documentation files written in clean Markdown.

| Document Filename | Domain / Subject | Key Policies & Facts |
| :--- | :--- | :--- |
| `account_security.md` | Authentication & Password Policies | - Max 3 password reset attempts per hour<br>- 60-minute lockout on exceed<br>- Password reset link validity: 15 min<br>- Password length: min 12 chars<br>- Mandatory TOTP 2FA for financial transactions<br>- 24h asset transfer hold after credential update |
| `api_rate_limits.md` | API Throttling & Developer Policies | - Standard Tier: 60 RPM, 1,000 RPH<br>- Enterprise Tier: 600 RPM<br>- Rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`<br>- HTTP 429 status code handling & exponential backoff with jitter |
| `kyc_verification.md` | Identity Verification (KYC / AML) | - Tier 1: ID card/passport + liveness selfie (automated, 15 min)<br>- Tier 2: Proof of Address within 90 days (manual, 24-48 business hours)<br>- **Strict Refusal Rule**: Support personnel cannot manually bypass, override, or expedite KYC for any user (including VIP accounts) |
| `payment_and_withdrawals.md` | Payments & Settlement Timelines | - Supported rails: Credit card, Wire transfer, Crypto (USDC/USDT)<br>- Fiat withdrawal SLA: 1-3 business days<br>- Crypto withdrawal SLA: 30 minutes after network confirmations<br>- Minimum withdrawal: $50 (Wire), $10 (Crypto)<br>- 24h withdrawal hold upon password reset |
| `incident_escalation.md` | Operational Incident & Escalation Policies | - Severity P1 (Critical Outage): 15-minute response SLA<br>- Severity P2 (Major Degradation): 30-minute response SLA<br>- Severity P3: 4-hour SLA, Severity P4: 24-hour SLA<br>- PagerDuty secondary escalation trigger: 10 minutes<br>- Post-Incident Review (PIR) SLA: 72 hours for P1/P2 |

---

## 3. Retrieval Architecture: Hybrid Vector + Lexical Search

### 3.1 Dense Vector Database (FAISS)
- **Engine**: FAISS (`faiss-cpu`)
  - **Index Structure**: `faiss.IndexFlatIP` (Inner Product on $L_2$-normalized vectors $\rightarrow$ exact Cosine Similarity).
  - **Metric**: Cosine Similarity in $[-1.0, 1.0]$.
  - **Index Persistence**: Local disk serialization (`vector_store/index.faiss` & metadata store `vector_store/metadata.pkl`).
- **Embedding Model**: OpenAI `text-embedding-3-small`
  - **Dimensionality**: 1536 dimensions.
  - **Normalization**: $L_2$ vector normalization enabled.
  - **Chunking Profile**:
    - Chunk Size: ~500 characters / ~100 tokens.
    - Chunk Overlap: 100 characters.
    - Preserved Metadata: `doc_id` (file name), `chunk_id`, `char_start`, `char_end`, `source_path`.
- **Resilience / Fallback Strategy**:
  - Deterministic MD5 feature-hashed normalized vectors when `OPENAI_API_KEY` is not present, guaranteeing offline test execution without external dependencies.

### 3.2 Sparse Lexical Index (Okapi BM25)
- **Algorithm**: Robertson-Spärck Jones Okapi BM25 ($k_1=1.5, b=0.75$).
- **Role**: High-precision lexical matching for technical acronyms (`KYC`, `TOTP`, `P1`, `PIR`), error codes (`HTTP 429`), and numerical metrics.

### 3.3 Reciprocal Rank Fusion (RRF) & Cross-Modal Reranker
- **Fusion Formula**:
  $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **Reranker Scoring**:
  $$\text{Score}_{\text{hybrid}} = (0.65 \times \text{Score}_{\text{dense}} + 0.35 \times \text{Score}_{\text{sparse\_norm}}) \times \text{AgreementBonus}$$
  where $\text{AgreementBonus} = 1.15$ if candidate was retrieved in both channels.

---

## 4. Evaluation Dataset Specification (`questions.json`)

The evaluation suite contains 8 test questions: 5 answerable and 3 unanswerable (out-of-bounds or policy absence).

| ID | Category | Question Text | Target Document | Expected Ground Truth / Refusal |
| :--- | :--- | :--- | :--- | :--- |
| `q1` | Answerable | How many password reset attempts are allowed per hour? | `account_security.md` | 3 attempts per hour before a 60-minute lockout. |
| `q2` | Answerable | What HTTP status code and response headers indicate an API rate limit has been exceeded? | `api_rate_limits.md` | HTTP 429; `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`. |
| `q3` | Answerable | How long does standard Tier 2 identity verification take to be reviewed? | `kyc_verification.md` | 24 to 48 business hours (manual review). |
| `q4` | Answerable | Is there any withdrawal hold after updating an account password or email? | `payment_and_withdrawals.md` | Yes, a mandatory 24-hour security hold. |
| `q5` | Answerable | What is the response SLA and PIR timeline for a P1 critical outage? | `incident_escalation.md` | 15-minute response SLA; PIR within 72 hours. |
| `q6` | Unanswerable | Can users pay an expedite fee of $50 to speed up Tier 2 KYC verification within 1 hour? | None (Unsupported) | Refusal: No expedite fees or 1-hour fast-track options exist in policy. |
| `q7` | Unanswerable | Can customer support issue full refunds for cryptocurrency transactions sent to an incorrect blockchain address? | None (Unsupported) | Refusal: Cryptocurrency transactions to incorrect addresses cannot be refunded. |
| `q8` | Unanswerable | What discount percentage is offered to customers who prepay for annual Enterprise API subscriptions? | None (Unsupported) | Refusal: Policy documentation does not state annual discount percentages. |

---

## 5. Artifact Schemas

### 5.1 Retrieval Output (`retrieval_results.json`)
```json
[
  {
    "question_id": "q1",
    "retrieved_chunks": [
      {
        "doc_id": "account_security.md",
        "chunk_id": "account_security.md#0",
        "score": 0.884,
        "text": "Users are permitted a maximum of 3 password reset attempts per hour..."
      }
    ]
  }
]
```

### 5.2 Answers Output (`answers.json`)
```json
[
  {
    "question_id": "q1",
    "answer": "Users are allowed a maximum of 3 password reset attempts per hour. If this limit is exceeded, the account is temporarily locked for 60 minutes.",
    "citations": ["account_security.md"],
    "supported": true
  },
  {
    "question_id": "q6",
    "answer": "The retrieved documentation does not provide any policy or provision for paying an expedite fee to speed up Tier 2 KYC verification.",
    "citations": [],
    "supported": false
  }
]
```

### 5.3 Validation Report (`validation_report.json`)
```json
[
  {
    "question_id": "q1",
    "is_valid": true,
    "checks": {
      "non_empty_answer": true,
      "citations_present_if_supported": true,
      "citations_in_retrieved_docs": true,
      "supported_flag_consistent": true
    },
    "details": "All deterministic checks passed."
  }
]
```

---

## 6. Deterministic Validation Rules

The validator implements four mandatory programmatic gates:
1. **Rule 1 (Non-Empty Response)**: `len(answer.strip()) > 0`.
2. **Rule 2 (Citation Enforcement for Supported Claims)**: If `supported == True`, then `len(citations) >= 1`.
3. **Rule 3 (Source Grounding / Provenance)**: For all $c \in \text{citations}$, $c \in \{chunk.\text{doc\_id} \mid chunk \in \text{retrieved\_chunks}\}$.
4. **Rule 4 (Refusal Consistency)**: If `supported == False`, citations must either be empty or explicitly flagged as ungrounded, and the answer must indicate insufficient context or refusal.
5. **Rule 5 (Retrieval Score Threshold - Stretch)**: If top retrieval similarity is below minimum threshold ($\tau = 0.40$), the pipeline flags the question as inherently unsupported.

---

## 7. Interfaces & API Contract

### CLI Interface
```bash
python app.py --question "How many password reset attempts are allowed per hour?"
```
**CLI Output:**
```json
{
  "question": "How many password reset attempts are allowed per hour?",
  "answer": "Users are permitted a maximum of 3 password reset attempts per hour...",
  "citations": ["account_security.md"],
  "supported": true,
  "retrieved_sources": [
    {
      "doc_id": "account_security.md",
      "chunk_id": "account_security.md#0",
      "score": 0.884
    }
  ]
}
```

### REST API Interface (FastAPI)
- **Endpoint**: `POST /ask`
- **Request Body**:
  ```json
  {
    "question": "string",
    "top_k": 3
  }
  ```
- **Response Body**:
  ```json
  {
    "question": "string",
    "answer": "string",
    "citations": ["string"],
    "supported": true,
    "retrieved_sources": [
      {
        "doc_id": "string",
        "chunk_id": "string",
        "score": 0.88
      }
    ]
  }
  ```

---

## 8. Observability & Logging Specification

- Pipeline events are emitted with structured timestamps and severity levels to both standard console and `pipeline.log`.
- Stage markers logged:
  - `[INGESTION]` Documents loaded & chunk counts.
  - `[INDEXING]` Vector embeddings computed & FAISS index initialized.
  - `[RETRIEVAL]` Query similarity execution, top chunk IDs & scores.
  - `[GENERATION]` Context assembly, token usage, LLM execution time.
  - `[VALIDATION]` Deterministic check results (Passed / Failed per rule).
