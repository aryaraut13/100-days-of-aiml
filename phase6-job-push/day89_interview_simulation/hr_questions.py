# day89_interview_simulation/hr_questions.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("[HR INTERVIEW PREPARATION]\n")

hr_qa = [
    {
        "q": "Tell me about yourself.",
        "a": "I am a co-founder and AI Engineer who has spent the last 89 days building AI products in public. I built 3 production-ready AI applications: a RAG support bot, a market research agent, and a product research agent. Before that, I studied engineering and worked on the technical side of my startup. I post daily on LinkedIn and commit daily to GitHub — consistency is something I take seriously."
    },
    {
        "q": "Why do you want to work here specifically?",
        "a": "I researched your product [X] and noticed you are building [Y with Z approach]. I have built something directly relevant — my RAG pipeline handles [similar problem]. I want to take what I have learned from building in public and apply it to a larger scale problem with a team. I can contribute from day one because I have the hands-on experience, not just theoretical knowledge."
    },
    {
        "q": "What is your biggest weakness?",
        "a": "I sometimes spend too long optimizing code before shipping. I have gotten better at this — my 100-day challenge forced me to ship something every single day, even when it was not perfect. That discipline has improved my ability to separate good enough from perfect."
    },
    {
        "q": "Where do you see yourself in 3 years?",
        "a": "Building AI products at scale and leading technical direction on LLM applications. I want to go deep on production AI systems — evaluation, reliability, cost optimization. I am not interested in switching domains. AI engineering is where I am going all in."
    },
    {
        "q": "Do you have any questions for us?",
        "a": "Three questions: What does the AI stack look like currently and what are the biggest technical challenges? How does the team evaluate LLM outputs in production? What would a successful first 90 days look like for this role?"
    },
]

for i, item in enumerate(hr_qa, 1):
    print(f"Q{i}: {item['q']}")
    print(f"A:  {item['a']}\n")
    print("-" * 60 + "\n")

print("[TIPS FOR HR ROUND]")
tips = [
    "Always connect answers back to specific projects you built",
    "Have 3 numbers ready: 0.78 overlap score, 100% fine-tune accuracy, 60 seconds for market report",
    "Ask about their AI stack in every interview — shows genuine interest",
    "Never say 'I am passionate about AI' — show it with GitHub commits",
    "Follow up with a thank you email within 24 hours",
]
for tip in tips:
    print(f"  * {tip}")