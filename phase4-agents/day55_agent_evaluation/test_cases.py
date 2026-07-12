TEST_CASES = [
    {
        "id": "TC001",
        "input": "What is the price of a laptop?",
        "expected_keywords": ["laptop", "price", "Rs"],
        "should_use_tool": True,
        "tool_expected": "get_product_price",
        "category": "product_lookup"
    },
    {
        "id": "TC002",
        "input": "Compare gaming laptop vs office laptop",
        "expected_keywords": ["gaming", "office", "price", "rating"],
        "should_use_tool": True,
        "tool_expected": "compare_products",
        "category": "comparison"
    },
    {
        "id": "TC003",
        "input": "What is the weather today?",
        "expected_keywords": ["don't have", "not available", "unable"],
        "should_use_tool": False,
        "tool_expected": None,
        "category": "out_of_scope"
    },
    {
        "id": "TC004",
        "input": "Recommend audio products under Rs.3000",
        "expected_keywords": ["headphone", "earbud", "Rs", "recommend"],
        "should_use_tool": True,
        "tool_expected": "get_recommendations",
        "category": "recommendation"
    },
    {
        "id": "TC005",
        "input": "Hello how are you?",
        "expected_keywords": ["hello", "help", "assist"],
        "should_use_tool": False,
        "tool_expected": None,
        "category": "greeting"
    },
]