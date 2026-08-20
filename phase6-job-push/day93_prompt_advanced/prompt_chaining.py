# day93_prompt_advanced/prompt_chaining.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=400
)


def prompt_chain(topic: str) -> dict:
    """
    Chain of prompts where each output feeds into the next.
    Step 1: Extract key facts
    Step 2: Identify gaps in knowledge
    Step 3: Generate questions to fill gaps
    Step 4: Synthesize final answer
    """
    print(f"[PROMPT CHAIN] Topic: {topic}\n")

    # Step 1
    facts = llm.invoke([HumanMessage(
        content=f"List 5 key facts about '{topic}' in bullet points. Facts only, no explanations."
    )]).content
    print(f"[STEP 1 - FACTS]\n{facts}\n")

    # Step 2 — uses Step 1 output
    gaps = llm.invoke([HumanMessage(
        content=f"Given these facts about {topic}:\n{facts}\n\nWhat are 3 important aspects NOT covered? List only the gaps."
    )]).content
    print(f"[STEP 2 - GAPS]\n{gaps}\n")

    # Step 3 — uses Step 2 output
    questions = llm.invoke([HumanMessage(
        content=f"Convert these knowledge gaps into 3 specific research questions:\n{gaps}"
    )]).content
    print(f"[STEP 3 - QUESTIONS]\n{questions}\n")

    # Step 4 — synthesizes everything
    synthesis = llm.invoke([HumanMessage(
        content=f"""Create a comprehensive 3-sentence summary of {topic} that:
1. Covers the key facts
2. Acknowledges the gaps
3. Points to areas needing more research

Facts: {facts}
Gaps: {gaps}"""
    )]).content
    print(f"[STEP 4 - SYNTHESIS]\n{synthesis}\n")

    return {
        "topic":     topic,
        "facts":     facts,
        "gaps":      gaps,
        "questions": questions,
        "synthesis": synthesis
    }


def self_consistency(question: str, samples: int = 3) -> str:
    """
    Self-consistency: run the same question multiple times
    and pick the most common answer. Reduces hallucination.
    """
    print(f"\n[SELF-CONSISTENCY] Question: {question}")
    print(f"Running {samples} independent samples...\n")

    answers = []
    for i in range(samples):
        response = llm.invoke([HumanMessage(
            content=f"Answer this question with a single number or short phrase only:\n{question}"
        )])
        answer = response.content.strip()
        answers.append(answer)
        print(f"  Sample {i+1}: {answer}")

    # Find most common answer
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0]
    print(f"\nMost consistent answer: '{most_common[0]}' (appeared {most_common[1]}/{samples} times)")
    return most_common[0]


if __name__ == "__main__":
    print("[ADVANCED PROMPTING TECHNIQUES]\n")
    print("=" * 60)

    # Prompt chaining
    prompt_chain("RAG (Retrieval Augmented Generation)")

    print("=" * 60)
    # Self-consistency
    self_consistency("How many parameters does GPT-3 have?", samples=3)
    self_consistency("What year was the transformer architecture introduced?", samples=3)