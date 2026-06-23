import os
import json
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


def extract_product_info(description: str) -> dict:
    """Extract structured product info from unstructured description."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract product information from the description.
Return ONLY valid JSON with these fields:
{{
  "product_name": "string",
  "category": "string",
  "price_inr": number or null,
  "key_features": ["list", "of", "features"],
  "target_audience": "string",
  "in_stock": true or false
}}
No text outside the JSON."""),
        ("human", "{description}")
    ])

    parser = JsonOutputParser()
    chain  = prompt | llm | parser
    return chain.invoke({"description": description})


if __name__ == "__main__":
    descriptions = [
        "The ProMax X1 wireless headphones are perfect for music lovers. At Rs.4999, they offer 30-hour battery life, active noise cancellation, and work with all Bluetooth devices. Currently available.",
        "Our premium yoga mat (Rs.1299) is made from eco-friendly TPE material, 6mm thick, non-slip surface. Ideal for beginners and advanced practitioners. Out of stock currently.",
    ]

    print("[STRUCTURED EXTRACTION]\n")
    for desc in descriptions:
        print(f"Input: {desc[:70]}...")
        result = extract_product_info(desc)
        print(f"Output:")
        print(json.dumps(result, indent=2))
        print()