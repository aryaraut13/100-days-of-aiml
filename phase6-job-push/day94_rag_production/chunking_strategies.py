# day94_rag_production/chunking_strategies.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

print("[CHUNKING STRATEGIES — PRODUCTION RAG]\n")

# Sample document
document = """
ECOMMERCE STORE — COMPLETE POLICY GUIDE

SECTION 1: RETURNS AND REFUNDS
Our return policy allows customers to return products within 30 days of purchase.
Items must be in original condition with all packaging intact. Damaged or used items
cannot be returned unless defective. Refunds are processed within 5-7 business days
after we receive the returned item. Digital products and personalized items are
non-refundable. To initiate a return, visit our returns portal or contact support.

SECTION 2: SHIPPING AND DELIVERY
Standard shipping takes 3-5 business days and is free on orders above Rs.500.
Express shipping takes 1-2 business days and costs Rs.99 extra. International
shipping is available to 15 countries with delivery in 7-14 days. Orders above
Rs.2000 qualify for free express shipping. Tracking information is sent to your
registered email within 24 hours of dispatch.

SECTION 3: PAYMENT OPTIONS
We accept Visa, Mastercard, UPI, PayTM, Google Pay, and net banking. Cash on
Delivery is available for orders under Rs.2000 in select pin codes. EMI options
are available through HDFC, ICICI, and SBI cards on orders above Rs.3000.
Buy Now Pay Later is available through LazyPay and ZestMoney partnerships.

SECTION 4: WARRANTY AND SUPPORT
All electronics come with a 1-year manufacturer warranty. Extended warranty of
2 years is available for Rs.299. Warranty claims require original invoice and
must be filed within the warranty period. Physical damage and water damage are
not covered. Contact support at support@store.com for warranty assistance.
"""

strategies = [
    {
        "name":    "Fixed Size (no overlap)",
        "splitter": CharacterTextSplitter(
            chunk_size=200, chunk_overlap=0, separator=" "
        )
    },
    {
        "name":    "Fixed Size (with overlap)",
        "splitter": CharacterTextSplitter(
            chunk_size=200, chunk_overlap=50, separator=" "
        )
    },
    {
        "name":    "Recursive (sentence-aware)",
        "splitter": RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
    },
]

for strategy in strategies:
    chunks = strategy["splitter"].split_text(document)
    avg_len = sum(len(c) for c in chunks) / len(chunks)

    print(f"[{strategy['name'].upper()}]")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: {avg_len:.0f} chars")
    print(f"  First chunk:    {chunks[0][:80]}...")
    print(f"  Last chunk:     {chunks[-1][:80]}...")
    print()

print("[WINNER FOR RAG]")
print("Recursive with overlap — preserves sentence boundaries")
print("and prevents answers from being cut mid-sentence.")
print("\nRule: chunk_size=300-500, chunk_overlap=50-100 for most RAG use cases.")