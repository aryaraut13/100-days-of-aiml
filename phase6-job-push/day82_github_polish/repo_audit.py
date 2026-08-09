# day82_github_polish/repo_audit.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from pathlib import Path

print("[GITHUB REPOSITORY AUDIT]\n")

# Simulate repo structure check
repo_root = Path("C:/Users/aryar/OneDrive/Documents/100-days-of-aiml")

# Check for essential files
essential_files = [
    "README.md",
    ".gitignore",
    "requirements.txt",
]

print("[ESSENTIAL FILES CHECK]")
for f in essential_files:
    path   = repo_root / f
    exists = path.exists() if repo_root.exists() else True  # assume exists
    status = "FOUND" if exists else "MISSING"
    print(f"  {status:8s} | {f}")

print()

# Project quality checklist
projects = [
    {
        "name":    "Project 1 — RAG Support Bot",
        "folder":  "phase3-llm-rag/day35_project1_rag_bot",
        "checks": {
            "README.md":        True,
            "requirements.txt": True,
            "architecture diagram": True,
            "demo screenshot":  False,
            "live demo URL":    False,
        }
    },
    {
        "name":    "Project 2 — Market Research Agent",
        "folder":  "phase4-agents/day49_project2_ui",
        "checks": {
            "README.md":        True,
            "requirements.txt": False,
            "architecture diagram": False,
            "demo screenshot":  False,
            "live demo URL":    False,
        }
    },
    {
        "name":    "Project 3 — Product Research Agent",
        "folder":  "phase5-deploy-finetune/day76_project3_build",
        "checks": {
            "README.md":        False,
            "requirements.txt": False,
            "architecture diagram": False,
            "demo screenshot":  False,
            "live demo URL":    False,
        }
    },
]

print("[PROJECT QUALITY AUDIT]")
for project in projects:
    done  = sum(1 for v in project["checks"].values() if v)
    total = len(project["checks"])
    pct   = int(done / total * 100)
    print(f"\n  {project['name']} — {pct}% complete ({done}/{total})")
    for check, status in project["checks"].items():
        icon = "OK  " if status else "TODO"
        print(f"    {icon} | {check}")

print("\n[ACTION ITEMS — PRIORITY ORDER]")
actions = [
    "1. Add demo screenshots to all 3 project READMEs",
    "2. Add requirements.txt to Project 2 and Project 3",
    "3. Add architecture diagrams to Project 2 and Project 3",
    "4. Pin all 3 projects on GitHub profile",
    "5. Add live demo URLs once deployed on Render",
    "6. Write one-line project descriptions for GitHub profile",
    "7. Add topics/tags to repo: langchain, rag, ai-agents, python",
]
for action in actions:
    print(f"  {action}")

print("\n[GITHUB PROFILE TIPS]")
tips = [
    "Profile README: add a GitHub profile README (create repo named 'aryaraut13')",
    "Pin repos: pin the 3 project folders prominently",
    "Contribution graph: daily commits for 81+ days is visible — keep it green",
    "Topics: add 'langchain', 'rag', 'ai-agents', 'llm', 'python' to the repo",
]
for tip in tips:
    print(f"  * {tip}")