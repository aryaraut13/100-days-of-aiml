# Project 2: Ecommerce Market Research Agent — Final Version

## Problem
Market research takes analysts days. The data is scattered across
multiple sources. This agent does it autonomously in under 60 seconds.

## What it does
Give it a product category. It autonomously:
1. Searches for top products with pricing and ratings
2. Analyzes price segments and market share
3. Gets market trends and CAGR data
4. Analyzes top competitors (strengths, weaknesses, positioning)
5. Searches and summarizes customer reviews
6. Writes a structured market research report

## Architecture
User Input (category + research type) | v [Supervisor Agent] <- Decides which tools to call and in what order | v [6 Custom Tools] <- Each tool returns structured data | v [Report Generation] <- Assembles findings into structured report | v Streamlit UI <- Displays formatted report
## Tools
| Tool | Purpose |
|------|---------|
| search_ecommerce_products | Find products by category/name |
| analyze_price_segments | Budget/mid/premium breakdown |
| get_market_trends | CAGR, key trends, opportunities |
| search_reviews | Customer sentiment analysis |
| analyze_competitor | Brand strengths and weaknesses |
| write_research_report | Structured report generation |

## Sample Output
Input: "Research the headphones market"
Output: Full market report with:
- Top 5 products with pricing and ratings
- Price segment breakdown (budget 45%, mid 35%, premium 20%)
- Market size Rs.3200 crore, 22% CAGR
- Competitor analysis: boAt vs Sony vs JBL
- Strategic opportunities identified

## Stack
LangChain Agents + LangGraph + Anthropic Claude + Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```