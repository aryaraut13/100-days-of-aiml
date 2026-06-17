# day31_langchain_basics/chains.py
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500
)


def sequential_chain():
    """
    Chain 1: topic → explanation
    Chain 2: explanation → quiz question
    Two LLM calls, output of first feeds into second.
    """
    explain_template = ChatPromptTemplate.from_messages([
        ("system", "You are a concise ML teacher."),
        ("human", "Explain {topic} in 3 sentences.")
    ])

    quiz_template = ChatPromptTemplate.from_messages([
        ("system", "You create quiz questions from explanations."),
        ("human", "Based on this explanation, write one multiple choice question with 4 options:\n\n{explanation}")
    ])

    explain_chain = explain_template | llm | StrOutputParser()
    quiz_chain    = quiz_template    | llm | StrOutputParser()

    # Chain them together
    full_chain = (
        {"explanation": explain_chain, "topic": RunnablePassthrough()}
        | quiz_chain
    )

    print("[SEQUENTIAL CHAIN]\n")
    print("Topic: Random Forest\n")

    explanation = explain_chain.invoke({"topic": "Random Forest"})
    print(f"Step 1 - Explanation:\n{explanation}\n")

    quiz = quiz_chain.invoke({"explanation": explanation})
    print(f"Step 2 - Quiz Question:\n{quiz}")


if __name__ == "__main__":
    sequential_chain()