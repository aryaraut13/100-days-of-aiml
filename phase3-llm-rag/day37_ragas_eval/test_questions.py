# day37_ragas_eval/test_questions.py
"""
Test dataset for RAGAS evaluation.
questions: what we ask
ground_truths: the correct answers (from our knowledge base)
"""

TEST_DATA = [
    {
        "question": "How do I return a product?",
        "ground_truth": "To return a product, visit our returns portal within 30 days of purchase. You will receive a full refund within 5-7 business days."
    },
    {
        "question": "What payment methods are accepted?",
        "ground_truth": "We accept Visa, Mastercard, UPI, PayTM, and net banking. Cash on Delivery is available for orders under Rs.2000."
    },
    {
        "question": "How long does standard shipping take?",
        "ground_truth": "Standard shipping takes 3-5 business days and is free on orders above Rs.500."
    },
    {
        "question": "Is EMI available?",
        "ground_truth": "EMI options are available on orders above Rs.3000 through partner banks. 0% EMI is available for 3 and 6 month tenures."
    },
    {
        "question": "How do I track my order?",
        "ground_truth": "Track your order using the tracking number sent to your registered email. The tracking number is sent within 24 hours of dispatch."
    },
]