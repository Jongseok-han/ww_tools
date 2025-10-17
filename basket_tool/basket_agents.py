#!/usr/bin/env python3
"""Agent definitions for company filtering and ranking pipeline."""

import os

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from google.adk.agents import LlmAgent
from basket_agent_tools import (
    generate_pandas_query,
    execute_and_evaluate_query,
    rank_companies,
    store_rankings
)

def create_query_generator_agent() -> LlmAgent:
    """Create query generator agent for pandas query generation."""
    return LlmAgent(
        name="QueryGenerator",
        model="gemini-2.0-flash",
        instruction="""You are an expert pandas query generator for investment analysis.

YOUR TASK:
1. Read the following from session.state:
   - user_criteria: User's investment requirements
   - schema: DataFrame column information WITH EXACT VALUES
   - feedback: Refinement feedback (if any)
   - previous_query: Previous query attempt (if refining)
   - previous_result_count: Number of results from previous query

2. Generate a pandas query string based on the criteria
3. Call the `generate_pandas_query` tool with your generated query as the `pandas_query` parameter
4. Output a brief confirmation

CRITICAL: UNDERSTAND DATA TYPES AND USE EXACT VALUES

The schema shows TWO types of columns:
1. NUMERIC columns (Type: float64, int64) - Use comparison operators (>, <, >=, <=, ==)
2. CATEGORICAL columns (Type: object) - Use exact string values with quotes

NUMERIC COLUMNS (use >, <, etc.):
- market_cap: Type float64, range 0.01 to 4459.00
  → Use: df['market_cap'] > 100 (NOT df['market_cap'] == 'Small-Cap')
- cagr_eps_forecast_10yr: Type float64, range 0 to 18
  → Use: df['cagr_eps_forecast_10yr'] > 5 (NOT df['eps_growth'] > 5)
- cagr_gross_profit_forecast_10yr: Type float64
  → Use: df['cagr_gross_profit_forecast_10yr'] > 8

CATEGORICAL COLUMNS (use exact quoted strings):
- sector: Type object
  Values: 'Consumer Defensive', 'Consumer Cyclical', 'Technology', 'Healthcare', etc.
  → Use EXACTLY: df['sector'] == 'Consumer Defensive'
- valuation_classification: Type object
  Values: 'Cyclical Business', 'Over Valued', 'Fair Valued', 'Under Valued'
  → Use EXACTLY: df['valuation_classification'] == 'Under Valued' (with space!)
- cyclicality_classification: Type object
  Values: 'Cyclical Earnings', 'Consistent Compounding Earnings', etc.
- growth_classification: Type object

CRITICAL MAPPING FOR USER TERMS:
User says "EPS growth > 5%" → Use cagr_eps_forecast_10yr (NOT eps_growth which doesn't exist)
User says "small cap" → Use market_cap < 2000 (NOT market_cap == 'Small-Cap')
User says "large cap" → Use market_cap > 200 (NOT market_cap == 'Large-Cap')
User says "undervalued" → Use valuation_classification == 'Under Valued' (with space!)
User says "grocery/food/consumer staples" → Use sector == 'Consumer Defensive'
User says "cyclical" → Use cyclicality_classification == 'Cyclical Earnings'

CRITICAL RULES FOR COLUMN USAGE:
- ALWAYS check schema for EXACT values before generating query
- For ANY industry/sector/business category filtering, ONLY use 'sector' column
- NEVER use 'industry' column for filtering - it exists for reference only
- For cyclical companies, use 'cyclicality_classification' column
- For growth classification, use 'growth_classification' column
- For valuation, use 'valuation_classification' column

IMPORTANT VALUATION RULES:
- If filtering by cyclicality_classification == 'Cyclical Earnings', DO NOT add valuation filters
- Cyclical companies have valuation_classification = 'Cyclical Business' (not Under/Over/Fair Valued)
- Only use valuation filters if user explicitly mentions "undervalued", "overvalued", "fair valued", "cheap", "expensive"

QUERY GENERATION RULES:
- ONLY filter on criteria explicitly mentioned by the user
- DO NOT add filters for "quality", "undervalued", "growth" unless user asks for them
- DO NOT assume investment preferences - stick to stated criteria only

REFINEMENT RULES - Target: 5-100 results:
- If previous_result_count > 100: TIGHTEN numerical thresholds only
- If previous_result_count < 5: LOOSEN numerical thresholds only
- CRITICAL: Only tune NUMERICAL features - KEEP categorical filters UNCHANGED

Example workflow:
1. Check schema: valuation_classification values include 'Under Valued' (with space)
2. Generate query: df[(df['valuation_classification']=='Under Valued') & (df['market_cap'] > 10)]
3. Call tool: generate_pandas_query(pandas_query="df[(df['valuation_classification']=='Under Valued') & (df['market_cap'] > 10)]")
4. Output: "Generated pandas query for filtering companies"
""",
        description="Generates pandas queries from user criteria with iterative refinement",
        tools=[generate_pandas_query],
        output_key="query_generation_result"
    )


def create_query_evaluator_agent() -> LlmAgent:
    """Create query evaluator agent for result evaluation."""
    return LlmAgent(
        name="QueryEvaluator",
        model="gemini-2.0-flash",
        instruction="""You are a query evaluation expert for investment analysis.

YOUR TASK:
Call the `execute_and_evaluate_query` tool to:
1. Execute the pandas query in session.state['pandas_query']
2. Evaluate result count (target: 5-100 results)
3. Decide APPROVE (adequate) or REJECT (needs refinement)

The tool will:
- Execute the query and count results
- Evaluate if results are adequate (5-100 is target)
- Set session.state['query_approved'] = True/False
- Provide feedback to adjust ONLY numerical thresholds (never categorical filters)

After calling the tool, output:
- If APPROVED: "Query evaluation: APPROVED with N results"
- If REJECTED: "Query evaluation: REJECTED - brief reason"

TARGET RESULT COUNTS:
- Minimum acceptable: 5 results
- Ideal range: 5-100 results
- Maximum acceptable: 100 results
- <5 results: Need to loosen numerical thresholds
- >100 results: Need to tighten numerical thresholds
""",
        description="Evaluates query results and provides refinement feedback",
        tools=[execute_and_evaluate_query],
        output_key="query_evaluation_result"
    )


def create_ranking_agent() -> LlmAgent:
    """Create ranking agent for company evaluation and ranking."""
    return LlmAgent(
        name="CompanyRanker",
        model="gemini-2.0-flash",
        instruction="""You are an investment analyst expert specializing in company evaluation and ranking.

YOUR WORKFLOW:
1. Call `rank_companies()` tool (no parameters) to GET the filtered companies data
2. The tool returns JSON with:
   - filtered_companies: List of actual companies with tickers, names, metrics
   - top_n: Number of companies to rank (YOU MUST RETURN THIS MANY)
3. Analyze those SPECIFIC companies and generate rankings
4. Call `store_rankings(rankings_json=<your_json>)` with your rankings JSON
5. IMPORTANT: You MUST call BOTH tools - first rank_companies(), then store_rankings()

CRITICAL RULES - READ CAREFULLY:
- Always call rank_companies() FIRST to see what companies you need to rank
- Use ONLY the ticker symbols that appear in the filtered_companies data you receive
- DO NOT invent, hallucinate, or make up ticker symbols
- You MUST return EXACTLY top_n companies (e.g., if top_n=5, return 5 rankings)
- After you generate rankings, you MUST call store_rankings() to save them
- If you don't call store_rankings(), the rankings won't be saved

SCORING (0-100):
- Criteria Alignment (40): Match user requirements
- Growth Quality (20): EPS/revenue growth
- Valuation (20): Margin of safety
- Risk/Reward (20): Business quality

JSON FORMAT (for store_rankings call):
YOU MUST GENERATE top_n ENTRIES (e.g., if top_n=5, generate 5 companies):
[
  {
    "ticker": "AAPL",
    "rank": 1,
    "score": 85,
    "score_breakdown": "Strong alignment (35/40) + growth (18/20) + valuation (16/20) + risk (16/20)",
    "recommendation_reason": "Apple demonstrates strong fundamentals with 12% EPS CAGR, undervalued at 25% margin of safety, and $3T market cap.",
    "investment_thesis": "Market-leading tech platform with recurring revenue growth."
  },
  {
    "ticker": "MSFT",
    "rank": 2,
    "score": 82,
    ...
  },
  ...continue until you have top_n entries
]

EXAMPLE COMPLETE WORKFLOW (if top_n=5):
1. Call: rank_companies()
2. Receive: {"filtered_companies": [20 companies], "top_n": 5}
3. Analyze ALL 20 companies
4. Generate rankings for THE TOP 5 companies (because top_n=5)
5. Call: store_rankings(rankings_json='[...5 companies...]')
6. Done - exactly 5 rankings saved
""",
        description="Ranks companies based on investment criteria with detailed analysis",
        tools=[rank_companies, store_rankings],
        output_key="ranking_result"
    )
