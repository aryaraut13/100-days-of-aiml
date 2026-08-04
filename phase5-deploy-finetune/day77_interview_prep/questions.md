# AI Engineer Interview Questions — Day 77

## LLM and RAG
Q: What is RAG and why is it better than fine-tuning for factual questions?
A: RAG retrieves relevant documents at query time and feeds them as context.
   Fine-tuning bakes knowledge into weights. RAG is better for frequently
   changing data because you update the knowledge base not the model.

Q: What is hallucination and how do you reduce it?
A: Hallucination is when an LLM generates confident but incorrect information.
   Reduce it by: grounding responses in retrieved documents (RAG),
   using temperature=0 for factual tasks, adding verification steps,
   and instructing the model to say "I don't know" when uncertain.

Q: Explain the difference between semantic search and keyword search.
A: Keyword search matches exact words (BM25, TF-IDF).
   Semantic search matches meaning via embedding similarity.
   Hybrid search combines both for production use.

## Agents and Tools
Q: What is the ReAct pattern?
A: Reason + Act. The agent alternates between reasoning (thought) and
   taking action (tool call). Each observation updates the reasoning.
   This loop continues until the task is complete.

Q: What is the difference between a chain and an agent?
A: A chain follows a fixed sequence of steps (deterministic).
   An agent dynamically decides which tools to call and in what order
   based on the current state (non-deterministic).

Q: How do you handle agent failures in production?
A: Timeout limits on tool calls. Graceful error messages from tools.
   Fallback responses when tools fail. Maximum iteration limits.
   LangSmith tracing to debug failures post-hoc.

## System Design
Q: How would you design a RAG system for 1 million documents?
A: Use a hosted vector DB (Pinecone, Weaviate). Chunk documents at
   ingest time. Use batch embedding. Add a caching layer (Redis) for
   frequent queries. Monitor retrieval quality with RAGAS metrics.

Q: How do you reduce LLM API costs in production?
A: Response caching (50% cost reduction). Prompt compression.
   Use smaller models for classification tasks. Batch requests.
   Monitor token usage per endpoint.

## Fine-tuning
Q: When should you fine-tune vs use RAG?
A: Fine-tune for: style/format/tone changes, domain-specific vocabulary,
   tasks requiring consistent structured output.
   Use RAG for: factual knowledge that changes, large knowledge bases,
   cases where you need to cite sources.

Q: What is LoRA and why is it useful?
A: Low-Rank Adaptation. Adds small trainable adapter matrices to frozen
   model weights. Only 0.1-1% of parameters are trained.
   Makes fine-tuning feasible on consumer hardware.