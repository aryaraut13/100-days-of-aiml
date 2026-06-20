import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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
    ("system", "You are a helpful ML tutor. Keep answers brief - 2 sentences max."),
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


def chat(message: str, session_id: str = "user_1") -> str:
    return chain_with_history.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )


if __name__ == "__main__":
    print("[CONVERSATION WITH MEMORY]\n")

    conversation = [
        "What is overfitting in machine learning?",
        "How do I fix it?",
        "What about the opposite problem?",
        "Which of these is more common in practice?",
    ]

    for message in conversation:
        print(f"User: {message}")
        response = chat(message)
        print(f"Bot:  {response}\n")