"""
Ingest documents into ChromaDB vector store.
Run this once before starting the app.
"""
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FAQ_CONTENT = """
ECOMMERCE STORE - CUSTOMER FAQ

RETURNS & REFUNDS
To return a product, visit our returns portal within 30 days of purchase.
You will receive a full refund within 5-7 business days to your original payment method.
Items must be unused and in original packaging.

SHIPPING
Standard shipping takes 3-5 business days and is free on orders above Rs.500.
Express shipping takes 1-2 days and costs Rs.99 extra.
International shipping is available to 15 countries.

PAYMENTS
We accept Visa, Mastercard, UPI, PayTM, and net banking.
Cash on Delivery is available for orders under Rs.2000.
EMI options are available on orders above Rs.3000 through partner banks.
0% EMI is available for 3 and 6 month tenures.

ORDER TRACKING
Track your order using the tracking number sent to your registered email.
The tracking number is sent within 24 hours of dispatch.
You can also track orders from your account dashboard.

WARRANTY
All products come with a 1-year manufacturer warranty.
Extended warranty of 2 years is available for Rs.299.
Warranty claims can be filed at warranty@store.com.

SUPPORT
Contact our support team at support@store.com.
Call us at 1800-123-4567 (Monday to Saturday, 9am to 6pm IST).
Average response time is 2-4 hours on business days.

BULK ORDERS
Bulk orders of 10 or more items receive a 15% discount automatically.
For orders above Rs.50,000 contact our B2B team at b2b@store.com.
"""

with open("faq.txt", "w") as f:
    f.write(FAQ_CONTENT)

loader   = TextLoader("faq.txt")
docs     = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks   = splitter.split_documents(docs)

embeddings  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

print(f"[INGEST] {len(chunks)} chunks stored in ChromaDB")
print(f"         Persisted to ./chroma_db")
print(f"         Ready for the app.")