#!/usr/bin/env python3
"""
Company Filtering Pipeline - ADK-Powered Investment Analysis

Orchestrates company filtering and ranking workflow using Google ADK agents:
- LoopAgent: [QueryGenerator, QueryEvaluator] for iterative query refinement
- SequentialAgent: [RefinementLoop, Ranker] for end-to-end pipeline
"""

import os
import pandas as pd
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LoopAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from basket_agents import (
    create_query_generator_agent,
    create_query_evaluator_agent,
    create_ranking_agent
)
from basket_agent_tools import set_dataframe

@dataclass
class CompanyRecommendation:
    """Ranked company recommendation"""
    ticker: str
    company_name: str
    rank: int
    score: float
    score_breakdown: str
    recommendation_reason: str
    key_metrics: Dict[str, Any]
    investment_thesis: str


def generate_df_schema(df: pd.DataFrame) -> str:
    """Generate schema with column types, categorical values, and numeric statistics."""
    schema_info = []
    schema_info.append("=" * 80)
    schema_info.append("COMPANY DATASET SCHEMA")
    schema_info.append("=" * 80)
    schema_info.append(f"\nTotal Records: {len(df)}")
    schema_info.append(f"Total Columns: {len(df.columns)}\n")

    # Selected columns for analysis
    selected_cols = [
        'ticker', 'name', 'industry', 'sector', 'market_cap',
        'cagr_eps_forecast_10yr', 'cagr_gross_profit_forecast_10yr',
        'cyclicality_classification', 'price', 'target_price',
        'valuation_classification', 'growth_classification'
    ]

    # Filter to existing columns
    available_cols = [col for col in selected_cols if col in df.columns]

    schema_info.append("\nCOLUMN DETAILS:")
    schema_info.append("-" * 80)

    for col in available_cols:
        schema_info.append(f"\n📊 {col}")
        schema_info.append(f"   Type: {df[col].dtype}")
        schema_info.append(f"   Non-null Count: {df[col].notna().sum()}/{len(df)}")

        if df[col].dtype == 'object' or col in ['sector', 'cyclicality_classification',
                                                'valuation_classification', 'growth_classification']:
            unique_vals = df[col].dropna().unique()
            schema_info.append(f"   Unique Values ({len(unique_vals)}): {', '.join(map(str, unique_vals))}")
        elif pd.api.types.is_numeric_dtype(df[col]):
            schema_info.append(f"   Statistics:")
            schema_info.append(f"      Min: {df[col].min():.2f}")
            schema_info.append(f"      Max: {df[col].max():.2f}")
            schema_info.append(f"      Mean: {df[col].mean():.2f}")
            schema_info.append(f"      Median: {df[col].median():.2f}")
            schema_info.append(f"      25th percentile: {df[col].quantile(0.25):.2f}")
            schema_info.append(f"      75th percentile: {df[col].quantile(0.75):.2f}")

    schema_info.append("\n" + "=" * 80)
    schema_info.append("PANDAS QUERY EXAMPLES:")
    schema_info.append("=" * 80)
    schema_info.append("""
1. Filter by market cap and sector:
   df[(df['market_cap'] > 10e9) & (df['sector'] == 'Technology')]

2. Filter by valuation and growth:
   df[(df['valuation_classification'] == 'Undervalued') & (df['cagr_eps_forecast_10yr'] > 10)]

3. Filter by multiple criteria:
   df[(df['sector'] == 'Financial Services') &
      (df['market_cap'] > 5e9) &
      (df['valuation_classification'].isin(['Undervalued', 'Fair Value']))]

4. Filter by cyclicality:
   df[df['cyclicality_classification'].str.contains('Cyclical', na=False)]

5. Sort and filter:
   df[df['cagr_eps_forecast_10yr'] > 0].sort_values('cagr_eps_forecast_10yr', ascending=False).head(20)
""")

    return "\n".join(schema_info)


def find_investment_opportunities(
    df: pd.DataFrame,
    criteria: str,
    top_n: int = 10,
    max_iterations: int = 3
) -> List[CompanyRecommendation]:
    """Find investment opportunities using ADK agent pipeline with iterative query refinement."""
    print(f"\n{'='*100}")
    print(f"COMPANY FILTERING PIPELINE")
    print(f"{'='*100}")
    print(f"Criteria: {criteria}")
    print(f"Target: Top {top_n} companies")
    print(f"Max refinement iterations: {max_iterations}")
    print(f"{'='*100}\n")

    set_dataframe(df)
    schema = generate_df_schema(df)

    query_generator = create_query_generator_agent()
    query_evaluator = create_query_evaluator_agent()
    ranking_agent = create_ranking_agent()

    refinement_loop = LoopAgent(
        name="QueryRefinementLoop",
        sub_agents=[query_generator, query_evaluator],
        max_iterations=max_iterations,
        description="Iteratively refine query until adequate results (5-100 companies)"
    )

    pipeline = SequentialAgent(
        name="CompanyFilterPipeline",
        sub_agents=[refinement_loop, ranking_agent],
        description="Filter companies and rank by investment criteria"
    )

    session_service = InMemorySessionService()
    initial_state = {
        'user_criteria': criteria,
        'schema': schema,
        'feedback': '',
        'pandas_query': '',
        'previous_query': '',
        'previous_result_count': 0,
        'query_approved': False,
        'attempt_number': 1,
        'result_count': 0,
        'filtered_df': None,
        'filtered_companies': [],
        'top_n': top_n,
        'rankings': []
    }

    runner = Runner(
        agent=pipeline,
        app_name="CompanyFilterPipeline",
        session_service=session_service
    )

    async def run_pipeline():
        new_session = session_service.create_session(
            app_name="CompanyFilterPipeline",
            user_id="investor",
            state=initial_state
        )
        if asyncio.iscoroutine(new_session):
            new_session = await new_session

        from google.genai import types
        content = types.Content(
            role="user",
            parts=[types.Part(text=f"Find and rank companies matching: {criteria}")]
        )

        async for event in runner.run_async(
            user_id="investor",
            session_id=new_session.id,
            new_message=content
        ):
            pass  # Tools print progress

        final_session = session_service.get_session(
            app_name="CompanyFilterPipeline",
            user_id="investor",
            session_id=new_session.id
        )
        if asyncio.iscoroutine(final_session):
            final_session = await final_session

        return final_session.state

    # Handle Jupyter notebook event loops
    try:
        asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
    except RuntimeError:
        pass

    final_state = asyncio.run(run_pipeline())

    rankings = final_state.get('rankings', [])
    result_count = final_state.get('result_count', 0)
    iterations = final_state.get('attempt_number', 1) - 1

    print(f"\n{'='*100}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*100}")
    print(f"Query refinement iterations: {iterations}")
    print(f"Filtered companies: {result_count}")
    print(f"Ranked recommendations: {len(rankings)}")
    print(f"{'='*100}\n")

    recommendations = []
    for rank_data in rankings:
        recommendations.append(CompanyRecommendation(
            ticker=rank_data.get('ticker', ''),
            company_name=rank_data.get('company_name', 'N/A'),
            rank=rank_data.get('rank', 0),
            score=rank_data.get('score', 0),
            score_breakdown=rank_data.get('score_breakdown', 'N/A'),
            recommendation_reason=rank_data.get('recommendation_reason', ''),
            key_metrics=rank_data.get('key_metrics', {}),
            investment_thesis=rank_data.get('investment_thesis', '')
        ))

    return recommendations


def display_recommendations(recommendations: List[CompanyRecommendation]):
    """Display recommendations in formatted output"""
    if not recommendations:
        print("\nNo recommendations found")
        return

    print(f"\n{'='*100}")
    print(f"TOP COMPANY RECOMMENDATIONS")
    print(f"{'='*100}\n")

    for rec in recommendations:
        print(f"#{rec.rank} - {rec.ticker} ({rec.company_name}) - Score: {rec.score}/100")
        print(f"{'-'*100}")
        print(f"Score Breakdown: {rec.score_breakdown}")
        print(f"\nRecommendation: {rec.recommendation_reason}")
        print(f"\nInvestment Thesis: {rec.investment_thesis}")
        print(f"\nKey Metrics:")
        for metric, value in rec.key_metrics.items():
            if value is not None:
                print(f"  • {metric}: {value}")
        print()
