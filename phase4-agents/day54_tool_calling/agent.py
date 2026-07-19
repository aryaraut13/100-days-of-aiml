import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1000
)

# Web search tool
search = TavilySearchResults(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY")
)

tools = [search]
agent = create_react_agent(llm, tools)


def ask(question: str) -> str:
    result = agent.invoke({"messages": [("human", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("[WEB SEARCH AGENT — REAL-TIME INFORMATION]\n")

    questions = [
        "What are the latest AI models released in 2025?",
        "What is the current price of a boAt Rockerz 450 on Amazon India?",
        "What are the top Python libraries for AI agents in 2025?",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer = ask(q)
        print(f"A: {answer[:200]}...\n")
        print("-" * 55)