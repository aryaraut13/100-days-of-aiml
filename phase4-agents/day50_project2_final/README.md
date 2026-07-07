# Ecommerce Market Research Agent

An autonomous AI agent that researches ecommerce markets, analyzes competitors,
and generates structured market intelligence reports.

## What it does
Give it a product category. It autonomously:
1. Searches for top products with pricing and ratings
2. Analyzes price segments and market share
3. Gets market trends and growth data
4. Analyzes top competitors (strengths, weaknesses, positioning)
5. Searches and summarizes customer reviews
6. Writes a structured market research report

## Stack
- **LangChain Agents** — ReAct agent with tool routing
- **LangGraph** — agent execution framework
- **Anthropic Claude** — language model
- **Streamlit** — web UI
- Custom tools — product search, competitor analysis, report writing

## Tools
| Tool | Purpose |
|------|---------|
| search_ecommerce_products | Find products by category/name |
| analyze_price_segments | Budget/mid/premium breakdown |
| get_market_trends | CAGR, key trends, opportunities |
| search_reviews | Customer sentiment analysis |
| analyze_competitor | Brand strengths and weaknesses |
| write_research_report | Structured report generation |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample output
Input: "Research the headphones market"
Output: Full market report with price segments, competitor analysis,
        growth trends, and strategic recommendations