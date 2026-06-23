import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


# Define the output schema with Pydantic
class ProductReview(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    score: float = Field(description="sentiment score from 0.0 to 1.0")
    key_issues: List[str] = Field(description="list of main issues or praises mentioned")
    summary: str = Field(description="one sentence summary of the review")
    recommended: bool = Field(description="whether the reviewer recommends the product")


class ReviewBatch(BaseModel):
    reviews: List[ProductReview]
    overall_sentiment: str
    average_score: float


parser = JsonOutputParser(pydantic_object=ProductReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a review analysis API.
Analyse the given product review and return ONLY valid JSON matching this schema:
{format_instructions}

Do not include any text outside the JSON."""),
    ("human", "Analyse this review: {review}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser


def analyse_review(review: str) -> dict:
    return chain.invoke({"review": review})


if __name__ == "__main__":
    reviews = [
        "This laptop is absolutely incredible! Battery lasts all day, keyboard feels great. Best purchase of the year. Highly recommend!",
        "Terrible product. Broke after 2 weeks. Customer service was useless and refused to help. Complete waste of money.",
        "It's decent. Does what it says. Nothing special but nothing terrible either. Average product for the price.",
    ]

    print("[PYDANTIC OUTPUT PARSER]\n")
    for review in reviews:
        print(f"Review: {review[:60]}...")
        result = analyse_review(review)
        print(f"  Sentiment:   {result['sentiment']}")
        print(f"  Score:       {result['score']}")
        print(f"  Recommended: {result['recommended']}")
        print(f"  Key issues:  {result['key_issues']}")
        print(f"  Summary:     {result['summary']}")
        print()