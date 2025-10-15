#!/usr/bin/env python3
"""
News Evaluation Agent - LLM-Powered News Relevance Analysis

Uses Google Gemini to evaluate news articles based on custom guidelines and
returns top 5 most relevant articles with reasoning.
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import google.generativeai as genai
from dotenv import load_dotenv

# Suppress gRPC ALTS warnings (not running on GCP)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

load_dotenv()

@dataclass
class EvaluatedNews:
    """Evaluated news article with LLM reasoning"""
    title: str
    summary: str
    url: str
    publish_date: str
    source: str
    rank: int  # Comparative rank (1 = most relevant)
    evaluation_reason: str
    key_insights: List[str]


class NewsEvaluationAgent:
    """LLM-powered news evaluator using Google Gemini"""

    DEFAULT_GUIDELINES = """
    You are evaluating news articles to find ACTUAL NEWS STORIES, not financial dashboards or data pages.

    **CRITICAL: REJECT THESE IMMEDIATELY (Not News):**
    ❌ **Stock Quote/Dashboard Pages** - ANY article that is primarily:
       - Just stock price, charts, and financial metrics (EPS, Revenue, P/E ratio)
       - Content with ONLY numbers and no narrative/analysis/opinion/story
       - Auto-generated financial data aggregations without human analysis
       - Generic company profiles with historical data but no recent events

    ❌ **Earnings Calendar Pages** - Just lists of upcoming earnings dates without analysis
    ❌ **Financial Data Tables** - Purely data tables (balance sheet, income statement) without commentary
    ❌ **Stock Screener Results** - Lists of stocks matching criteria without individual analysis

    **WHAT QUALIFIES AS REAL NEWS (Select Only These):**
    ✅ **Earnings ANALYSIS** (not just data):
       - Articles discussing WHY earnings beat/missed expectations
       - Management commentary on guidance changes
       - Analyst reactions to earnings with specific opinions

    ✅ **Product/Business DEVELOPMENTS** (actual events):
       - New product launches with details about features/markets
       - Partnership announcements with terms and strategic reasoning
       - M&A news with deal terms and strategic rationale
       - Executive changes with context about why and what it means

    ✅ **Analyst REPORTS** (with new insights):
       - Upgrade/downgrade with detailed justification
       - New research with original analysis and specific thesis
       - Must have analyst name, firm, and specific reasoning
       - NOT just "Price target raised to $X" without explanation

    ✅ **Market-Moving EVENTS** (newsworthy):
       - Regulatory actions with implications explained
       - Legal developments with context and potential impact
       - Competitive threats with analysis of market dynamics
       - Industry changes affecting the company specifically

    **NEWS QUALITY CHECKLIST - Article MUST have ALL of these:**
    1. ✅ **Narrative/Story** - Written by a journalist/analyst, not auto-generated
    2. ✅ **Attribution** - Named sources (analysts, executives, officials) or byline
    3. ✅ **Analysis** - Explains "why it matters" or "what it means", not just "what happened"
    4. ✅ **Recency** - Discusses recent events (within days), not just historical data
    5. ✅ **Actionable Insights** - Investors can learn something new or make decisions

    **STRICT FILTERING RULES:**
    - If article content is 80%+ numbers with <20% narrative → REJECT
    - If summary is just financial metrics without context → REJECT
    - When in doubt about whether it's news vs. data page → REJECT

    **RANKING PRIORITY (for articles that pass filtering):**
    1. Breaking news with material impact (earnings surprises, M&A, regulatory action)
    2. Analyst reports with new fundamental thesis and detailed reasoning
    3. Product launches or business developments with strategic implications
    4. Market-moving events with clear cause-and-effect explained
    5. Industry trends affecting the company with specific analysis
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the news evaluation agent"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ News Evaluation Agent initialized with {model_name}")

    def evaluate_news(
        self,
        articles: List[Any],
        ticker: str,
        company_name: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        top_k: int = 5
    ) -> List[EvaluatedNews]:
        """
        Evaluate news articles using LLM and return top K most relevant

        Args:
            articles: List of NewsArticle objects from financial_news.py
            ticker: Stock ticker symbol
            company_name: Optional company name for context
            custom_instructions: Optional user-defined focus for ranking. When provided, articles matching
                               this focus will be given HIGHEST PRIORITY in ranking, regardless of topic.
            top_k: Number of top articles to return (default: 5)

        Returns:
            List of EvaluatedNews objects with reasoning
        """
        if not articles:
            print("❌ No articles to evaluate")
            return []

        print(f"\n🤖 Evaluating {len(articles)} articles for {ticker}...")

        # Prepare articles data for LLM
        articles_data = []
        for i, article in enumerate(articles, 1):
            pub_date = article.publish_date.strftime('%Y-%m-%d %H:%M') if article.publish_date else 'Recent'
            articles_data.append({
                'id': i,
                'title': article.title,
                'summary': article.summary[:500] if article.summary else "No summary",
                'source': article.source,
                'publish_date': pub_date,
                'url': article.url
            })

        # Build evaluation prompt - combine default guidelines with custom if provided
        company_context = f" ({company_name})" if company_name else ""

        if custom_instructions:
            # Custom instructions given HIGHEST PRIORITY for ranking
            guidelines_text = f"""{self.DEFAULT_GUIDELINES}

🎯 **USER'S CUSTOM FOCUS - HIGHEST PRIORITY FOR RANKING:**
{custom_instructions}

**CRITICAL RANKING INSTRUCTION:**
The user's custom focus above is the MOST IMPORTANT factor for ranking.
- Articles matching this focus MUST be ranked higher than other articles
- The custom focus overrides the general ranking priorities listed earlier
- Even if an article is excellent but doesn't match the focus, it should be ranked lower than articles matching the focus
- Treat the custom focus as the primary lens through which to evaluate and order all news"""
        else:
            guidelines_text = self.DEFAULT_GUIDELINES

        prompt = f"""You are a financial news analyst evaluating news articles for {ticker}{company_context}.

EVALUATION GUIDELINES:
{guidelines_text}

ARTICLES TO EVALUATE:
{json.dumps(articles_data, indent=2)}

TASK:
First, FILTER OUT all non-news articles (dashboard pages, stock quotes, data tables).
Then, rank the REAL NEWS articles and return the top {top_k} in order of importance.

For each selected article, provide:
1. **Article ID** (from the list above)
2. **Rank** (1 = most important, 2 = second most, etc.)
3. **Evaluation Reason** (2-3 sentences explaining WHY this is real news and why it matters)
4. **Key Insights** (2-4 bullet points of NEW information or analysis, not just restating numbers)

Return your evaluation as a JSON array with this exact structure:
```json
[
  {{
    "article_id": 1,
    "rank": 1,
    "evaluation_reason": "This is actual news because [journalist/analyst name] reported that... The analysis explains...",
    "key_insights": [
      "Specific insight about what changed and why it matters",
      "Forward-looking implication for investors"
    ]
  }},
  {{
    "article_id": 5,
    "rank": 2,
    "evaluation_reason": "...",
    "key_insights": ["..."]
  }}
]
```

CRITICAL REMINDERS:
- REJECT articles with titles like "Stock Price Today", "Stock Quote", "Stock Chart"
- REJECT if URL is cnn.com/markets/stocks/, tradingview.com, investing.com/equities/, yahoo.com/quote/
- REJECT if content is just financial metrics without narrative/analysis
- Only select articles with NARRATIVE, ATTRIBUTION, and ANALYSIS
- Rank 1 = most impactful news story
- Return ONLY the JSON array, no additional text."""

        try:
            # Call Gemini API with adjusted settings for longer responses
            print(f"  🔄 Calling Gemini API for evaluation...")

            generation_config = {
                'temperature': 0.3,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 8192,  # Increased to handle large responses
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            response_text = response.text.strip()

            # Extract JSON from response
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end == -1:  # No closing ```
                    response_text = response_text[json_start:].strip()
                else:
                    response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                if json_end == -1:  # No closing ```
                    response_text = response_text[json_start:].strip()
                else:
                    response_text = response_text[json_start:json_end].strip()

            # Try to parse evaluation results
            try:
                evaluations = json.loads(response_text)
            except json.JSONDecodeError as e:
                # Try to recover incomplete JSON by finding the last complete object
                print(f"  ⚠️  JSON truncated, attempting to recover valid entries...")

                # Find last complete object by looking for closing }]
                last_complete = response_text.rfind('}')
                if last_complete != -1:
                    # Try adding closing bracket
                    recovered_text = response_text[:last_complete+1] + ']'
                    try:
                        evaluations = json.loads(recovered_text)
                        print(f"  ✅ Recovered {len(evaluations)} valid entries")
                    except:
                        raise e
                else:
                    raise e

            # Map evaluations back to original articles
            evaluated_articles = []
            for eval_item in evaluations[:top_k]:
                article_id = eval_item['article_id']
                if 1 <= article_id <= len(articles):
                    original_article = articles[article_id - 1]
                    pub_date = original_article.publish_date.strftime('%Y-%m-%d %H:%M') if original_article.publish_date else 'Recent'

                    evaluated_articles.append(EvaluatedNews(
                        title=original_article.title,
                        summary=original_article.summary or "No summary available",
                        url=original_article.url,
                        publish_date=pub_date,
                        source=original_article.source,
                        rank=eval_item['rank'],
                        evaluation_reason=eval_item['evaluation_reason'],
                        key_insights=eval_item['key_insights']
                    ))

            print(f"  ✅ Evaluation complete: {len(evaluated_articles)} articles selected")
            return evaluated_articles

        except json.JSONDecodeError as e:
            print(f"  ❌ Failed to parse LLM response as JSON: {e}")
            print(f"  Response text: {response_text[:500]}")
            return []
        except Exception as e:
            print(f"  ❌ Evaluation failed: {e}")
            return []

    def display_evaluation_results(self, evaluated_news: List[EvaluatedNews], ticker: str):
        """Display formatted evaluation results"""
        if not evaluated_news:
            print("No evaluated articles to display")
            return

        print(f"\n{'='*80}")
        print(f"📊 TOP {len(evaluated_news)} MOST RELEVANT NEWS FOR {ticker}")
        print(f"{'='*80}")

        for news in evaluated_news:
            print(f"\n#{news.rank} 📰 {news.title}")
            print(f"   {'─'*76}")
            print(f"   📅 Date: {news.publish_date} | 📺 Source: {news.source}")
            print(f"\n   💡 Why This Is Real News:")
            print(f"   {news.evaluation_reason}")
            print(f"\n   🎯 Key Insights:")
            for insight in news.key_insights:
                print(f"   • {insight}")
            print(f"\n   🔗 URL: {news.url}")
            print(f"   {'─'*76}")

    def save_evaluation_results(self, evaluated_news: List[EvaluatedNews], ticker: str, output_file: Optional[str] = None) -> str:
        """Save evaluation results to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"news_evaluation_{ticker}_{timestamp}.json"

        results = {
            'ticker': ticker,
            'evaluation_date': datetime.now().isoformat(),
            'total_articles_evaluated': len(evaluated_news),
            'articles': [asdict(news) for news in evaluated_news]
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Evaluation results saved: {output_file}")
        return output_file


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def evaluate_news_articles(
    articles: List[Any],
    ticker: str,
    company_name: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    top_k: int = 5,
    display_results: bool = True,
    save_results: bool = False
) -> List[EvaluatedNews]:
    """
    Convenience function to evaluate news articles with LLM

    Args:
        articles: List of NewsArticle objects from financial_news.py
        ticker: Stock ticker symbol
        company_name: Optional company name
        custom_instructions: Optional user-defined focus for ranking. Specify any topic, theme, or perspective
                           you want to prioritize. Articles matching this will be ranked HIGHEST.
        top_k: Number of top articles to return (default: 5)
        display_results: Whether to print formatted results (default: True)
        save_results: Whether to save results to JSON file (default: False)

    Returns:
        List of EvaluatedNews objects with LLM reasoning

    Example:
        # User can specify ANY focus based on their interests
        custom_focus = "I want to focus on [any topic the user cares about]"
        evaluated = evaluate_news_articles(articles, "AAPL", custom_instructions=custom_focus)

    """
    agent = NewsEvaluationAgent()

    evaluated_news = agent.evaluate_news(
        articles=articles,
        ticker=ticker,
        company_name=company_name,
        custom_instructions=custom_instructions,
        top_k=top_k
    )

    if display_results and evaluated_news:
        agent.display_evaluation_results(evaluated_news, ticker)

    if save_results and evaluated_news:
        agent.save_evaluation_results(evaluated_news, ticker)

    return evaluated_news


if __name__ == "__main__":
    from financial_news import get_news_for_ticker

    print("🧪 Testing News Evaluation Agent")
    ticker = "AAPL"
    articles = get_news_for_ticker(ticker, max_results=30, days_back=1)

    if articles:
        # Example 1: Standard evaluation (no custom focus)
        print("\n" + "="*80)
        print("TEST 1: Standard Evaluation (No Custom Focus)")
        print("="*80)
        evaluated_news = evaluate_news_articles(
            articles=articles,
            ticker=ticker,
            company_name="Apple Inc.",
            top_k=5,
            display_results=True,
            save_results=False
        )
        if evaluated_news:
            print(f"\n✅ Successfully evaluated {len(evaluated_news)} articles")

        # Example 2: With custom focus (user can specify ANY topic)
        print("\n" + "="*80)
        print("TEST 2: With Custom Focus (User-Defined Priority)")
        print("="*80)
        # Users can specify any focus based on their personal interests/needs
        custom_focus = "Focus on news that discusses long-term strategic implications"
        evaluated_news_custom = evaluate_news_articles(
            articles=articles,
            ticker=ticker,
            company_name="Apple Inc.",
            custom_instructions=custom_focus,
            top_k=5,
            display_results=True,
            save_results=True
        )
    else:
        print("❌ No articles retrieved")
