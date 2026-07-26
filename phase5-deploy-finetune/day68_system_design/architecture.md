# AI Application System Design

## Problem
Design a customer support AI system that handles 10,000 queries/day
with < 2 second response time and 99.9% uptime.

## Components

### 1. API Gateway
- Rate limiting per user (100 req/min)
- Authentication (API keys)
- Request routing
- Load balancing

### 2. Query Processing
- Input validation and sanitization
- Query classification (returns/shipping/payment/warranty)
- Cache check (Redis) -> if cached, return immediately
- Route to appropriate handler

### 3. RAG Pipeline
- Embed query with all-MiniLM-L6-v2
- Retrieve top-3 chunks from ChromaDB
- Assemble context + prompt
- Call LLM (Claude Haiku)
- Return grounded answer

### 4. Caching Layer (Redis)
- Cache frequent queries (TTL: 1 hour)
- Cache embeddings (TTL: 24 hours)
- Expected cache hit rate: 40-60%
- Reduces LLM calls and cost by ~50%

### 5. Monitoring
- Request count per endpoint
- P50/P95/P99 latency
- LLM token usage and cost
- Cache hit rate
- Error rate per category

## Capacity Estimation
- 10,000 queries/day = ~7 queries/minute (average)
- Peak: 3x average = 21 queries/minute
- LLM latency: ~1.5s average
- Cache hit saves ~1.2s per query
- Target: < 2s for 95th percentile

## Cost Estimation
- Claude Haiku: $0.00025 per 1K input tokens
- Average query: ~500 tokens input, 200 output
- Cost per query: ~$0.000175
- 10,000 queries/day: ~$1.75/day with 0% cache
- With 50% cache hit: ~$0.87/day

## Failure Modes
- LLM API down -> fallback to cached responses
- Vector DB slow -> timeout after 500ms, use keyword search
- Redis down -> bypass cache, serve directly
- All fail -> return "Service temporarily unavailable"