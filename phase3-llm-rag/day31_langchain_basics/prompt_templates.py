# day31_langchain_basics/prompt_templates.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=300
)


def basic_template():
    """Simple prompt template with variables."""
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains ML concepts simply."),
        ("human", "Explain {concept} in exactly 2 sentences for a beginner.")
    ])

    chain  = template | llm | StrOutputParser()

    concepts = ["gradient descent", "overfitting", "cross-validation"]
    print("[PROMPT TEMPLATES]\n")
    for concept in concepts:
        result = chain.invoke({"concept": concept})
        print(f"Concept: {concept}")
        print(f"Answer:  {result}\n")


def multi_variable_template():
    """Template with multiple variables."""
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}."),
        ("human", "{task}")
    ])

    chain = template | llm | StrOutputParser()

    print("[MULTI-VARIABLE TEMPLATE]\n")
    result = chain.invoke({
        "role": "senior data scientist reviewing ML model outputs",
        "task": "In one sentence, what is the most important metric for a fraud detection model and why?"
    })
    print(result)


if __name__ == "__main__":
    basic_template()
    multi_variable_template()