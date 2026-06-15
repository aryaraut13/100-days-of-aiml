from rag import rag_pipeline

if __name__ == "__main__":
    print("=" * 65)
    print("RAG PIPELINE — Ecommerce Customer Support Bot")
    print("=" * 65)

    questions = [
        "I bought something last week. How do I return it?",
        "What payment methods do you accept?",
        "My order hasn't arrived. How do I track it?",
        "Do you offer any discounts for large orders?",
        "What are your store opening hours?",
    ]

    for question in questions:
        rag_pipeline(question)
        print("-" * 65)