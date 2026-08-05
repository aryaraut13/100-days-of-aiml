# Mock Interview System Design — Day 79

## Question: Design a customer support AI system

### Clarifying questions to ask
- How many queries per day? (10K, 100K, 1M?)
- What is the acceptable latency? (< 1s, < 2s?)
- What is the knowledge base size? (100 docs, 10K docs?)
- Do we need multilingual support?
- What is the budget constraint?

### My answer (for 10K queries/day, < 2s latency)

**Architecture:**
User -> API Gateway -> Query Classifier -> [Cache Check] -> RAG Pipeline -> Response

**Components:**
1. API Gateway: rate limiting, auth, routing (FastAPI + Redis)
2. Query Classifier: categorize intent (returns/shipping/payment/warranty)
3. Cache: Redis with 1-hour TTL for frequent queries (50% hit rate target)
4. Embedding: HuggingFace all-MiniLM-L6-v2 (fast, good quality)
5. Vector DB: ChromaDB local / Pinecone hosted
6. LLM: Claude Haiku (fastest, cheapest, good quality)
7. Monitoring: LangSmith for traces, structured logging for metrics

**Cost estimate:**
- Claude Haiku: $0.00025/1K tokens
- 10K queries x 500 tokens = $1.25/day
- With 50% cache: $0.625/day
- Infrastructure: ~$20/month on Render free tier

**Scaling plan:**
- 10K/day: single instance, local ChromaDB
- 100K/day: add Redis cache, hosted vector DB, load balancer
- 1M/day: multiple instances, distributed cache, queue system

### Follow-up questions they might ask
Q: How do you handle the LLM being down?
A: Cache layer returns last known response. Fallback message for unknown queries.

Q: How do you improve answer quality over time?
A: Log all queries and answers. Review failed cases weekly.
   Update knowledge base with new FAQs. Re-evaluate with RAGAS metrics.

Q: How do you prevent prompt injection?
A: Input sanitization. System prompt sandboxing. Never execute user input as code.