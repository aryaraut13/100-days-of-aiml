import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#0f0f0f")
fig.patch.set_facecolor("#0f0f0f")

def box(ax, x, y, w, h, label, sublabel="", color="#5865F2"):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.1, label, ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.35, sublabel, ha="center", va="center",
                color="#cccccc", fontsize=8)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

# Title
ax.text(5, 9.3, "Ecommerce RAG Support Bot — Architecture",
        ha="center", va="center", color="white", fontsize=13, fontweight="bold")

# Boxes
box(ax, 5, 8.2, 3, 0.7, "User Question", color="#2d2d2d")
box(ax, 5, 7.0, 3.5, 0.7, "HuggingFace Embeddings", "all-MiniLM-L6-v2", "#764ba2")
box(ax, 5, 5.8, 3.5, 0.7, "ChromaDB Vector Store", "Semantic search -> Top 3 chunks", "#e67e22")
box(ax, 5, 4.6, 3.5, 0.7, "LangChain Chain", "LCEL pipe syntax", "#27ae60")
box(ax, 5, 3.4, 3.5, 0.7, "Claude Haiku", "Answer generation", "#e74c3c")
box(ax, 5, 2.2, 3, 0.7, "Grounded Answer", color="#2d2d2d")

# Side boxes
box(ax, 1.5, 5.8, 2, 0.6, "FAQ Documents", "Chunked + stored", "#2980b9")
box(ax, 8.5, 3.4, 2, 0.6, "FastAPI", "REST endpoint", "#16a085")
box(ax, 8.5, 4.6, 2, 0.6, "Streamlit", "Chat UI", "#8e44ad")
box(ax, 8.5, 2.2, 2, 0.6, "LangSmith", "Observability", "#f39c12")

# Arrows
arrow(ax, 5, 7.85, 5, 7.35)
arrow(ax, 5, 6.65, 5, 6.15)
arrow(ax, 5, 5.45, 5, 4.95)
arrow(ax, 5, 4.25, 5, 3.75)
arrow(ax, 5, 3.05, 5, 2.55)
arrow(ax, 2.5, 5.8, 3.25, 5.8)
arrow(ax, 7.5, 4.6, 7.5, 4.6)
arrow(ax, 6.75, 4.6, 7.5, 4.6)
arrow(ax, 6.75, 3.4, 7.5, 3.4)
arrow(ax, 6.75, 2.2, 7.5, 2.2)

plt.tight_layout()
plt.savefig("architecture.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f0f")
print("Architecture diagram saved -> architecture.png")