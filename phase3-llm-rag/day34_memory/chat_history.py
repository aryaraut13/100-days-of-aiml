import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=200
)

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


if __name__ == "__main__":
    print("[TWO SEPARATE SESSIONS - MEMORY IS ISOLATED]\n")

    print("=== Session 1 ===")
    r1 = chain_with_history.invoke(
        {"input": "My name is Arya and I am learning ML."},
        config={"configurable": {"session_id": "session_1"}}
    )
    print(f"Bot: {r1}\n")

    r2 = chain_with_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "session_1"}}
    )
    print(f"Bot: {r2}\n")

    print("=== Session 2 (fresh) ===")
    r3 = chain_with_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "session_2"}}
    )
    print(f"Bot: {r3}\n")

    print("[KEY INSIGHT]")
    print("Session 1 remembers the name. Session 2 has no context.")
    print("Memory is scoped per session - this is how production chat apps work.")