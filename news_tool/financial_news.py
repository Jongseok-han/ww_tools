#!/usr/bin/env python3
"""
Financial News Module - Multi-Source Retrieval

Sources (priority order): FMP → Tavily → Finnhub → AlphaVantage
Quality scores: Tavily(0.605), Benzinga/FMP(0.528), Zacks/AV(0.497)
Required: FMP_API_KEY, TAVILY_API_KEY, FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY in .env
"""

import os
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    url: str
    summary: str
    publish_date: Optional[datetime]
    source: str
    relevance_score: float = 0.0


class NewsRetriever:
    """Multi-source financial news retriever with relevance scoring"""

    FINANCIAL_KEYWORDS = [
        # === CORE FINANCIAL ===
        'stock', 'share', 'trading', 'price', 'market', 'earnings', 'revenue', 'profit', 
        'quarter', 'financial', 'analyst', 'investment', 'valuation', 'ipo', 'dividend',
        'sales', 'growth', 'margin', 'ebitda', 'cash flow', 'guidance', 'forecast', 'outlook',
        
        # === LEADERSHIP & GOVERNANCE ===
        'ceo', 'cfo', 'coo', 'president', 'chairman', 'board', 'executive', 'leadership',
        'management', 'director', 'founder', 'appointed', 'resigned', 'promoted', 'hired',
        'succession', 'governance', 'shareholder meeting', 'proxy',
        
        # === BUSINESS STRATEGY ===
        'merger', 'acquisition', 'partnership', 'joint venture', 'alliance', 'expansion',
        'launch', 'product', 'service', 'contract', 'deal', 'agreement', 'innovation',
        'technology', 'restructuring', 'reorganization', 'spinoff', 'divestiture',
        
        # === MARKET & COMPETITION ===
        'market share', 'competition', 'competitive', 'industry', 'sector', 'strategy',
        'customer', 'client', 'user base', 'subscriber', 'brand', 'reputation',
        'market leader', 'disruption', 'positioning', 'business model',
        
        # === REGULATORY & LEGAL ===
        'regulation', 'compliance', 'lawsuit', 'settlement', 'investigation', 'audit',
        'filing', 'disclosure', 'sec filing', 'fda approval', 'patent', 'license',
        'antitrust', 'litigation', 'court', 'fine',
        
        # === CORPORATE ACTIONS ===
        'buyback', 'repurchase', 'split', 'spin-off', 'restructure', 'bankruptcy',
        'offering', 'bond issuance', 'credit rating', 'announcement', 'press release',
        'conference call', 'webcast', 'milestone', 'achievement',
        
        # === PERFORMANCE METRICS ===
        'target', 'success', 'failure', 'progress', 'improvement', 'decline',
        'increase', 'decrease', 'surge', 'rally', 'volatility', 'momentum',
        'breakthrough', 'beat', 'miss', 'estimates', 'upgrade', 'downgrade',
        
        # === INDUSTRY & TECHNOLOGY ===
        'manufacturing', 'production', 'supply chain', 'digital transformation',
        'automation', 'ai', 'machine learning', 'cloud', 'software', 'platform',
        'e-commerce', 'capacity', 'distribution', 'ecosystem',
        
        # === EXTERNAL FACTORS ===
        'economic', 'recession', 'inflation', 'interest rates', 'policy', 'tariff',
        'trade war', 'global', 'currency', 'geopolitical', 'pandemic', 'recovery',
        
        # === STAKEHOLDERS ===
        'shareholder', 'investor', 'institutional', 'rating', 'recommendation',
        'price target', 'employee', 'layoff', 'hiring', 'workforce', 'compensation'
    ]

    TRUSTED_DOMAINS = [
        # Tier 1: Premium Financial News
        'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
        
        # Tier 2: Major Financial Media
        'cnbc.com', 'marketwatch.com', 'finance.yahoo.com', 'barrons.com',
        
        # Tier 3: Investment Analysis & Research
        'seekingalpha.com', 'motleyfool.com', 'investorplace.com', 'thestreet.com',
        'zacks.com', 'benzinga.com', 'morningstar.com', 'gurufocus.com',
        
        # Tier 4: Business News & General Media
        'forbes.com', 'fortune.com', 'businessinsider.com', 'cnn.com',
        'bbc.com', 'nytimes.com',
        
        # Tier 5: Financial Data & Trading Platforms
        'nasdaq.com', 'tradingview.com', 'finviz.com', 'tipranks.com',
        'stockanalysis.com', 'investing.com', 'fintel.io',
        
        # Tier 6: Official & Regulatory
        'sec.gov', 'investor.gov', 'finra.org',
        
        # Tier 7: Tech & Industry News
        'techcrunch.com', 'theverge.com', 'venturebeat.com', 'arstechnica.com',
        
        # Tier 8: Press Releases & Newswires
        'businesswire.com', 'prnewswire.com', 'globenewswire.com'
    ]

    def __init__(self, days_back: int = 7):
        self.fmp_key = os.getenv('FMP_API_KEY', '')
        self.tavily_key = os.getenv('TAVILY_API_KEY', '')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY', '')
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY', '')
        self.days_back = days_back

        configured = [k for k, v in [
            ("FMP", self.fmp_key), ("Tavily", self.tavily_key),
            ("Finnhub", self.finnhub_key), ("AlphaVantage", self.alphavantage_key)
        ] if v]

        if configured:
            print(f"✅ Active sources: {', '.join(configured)}")
        else:
            print("❌ No API keys found in .env file")

    def _calculate_relevance_score(self, title: str, summary: str, ticker: str) -> float:
        """Calculate relevance score: ticker matching + keyword frequency"""
        title_lower, summary_lower, ticker_lower = title.lower(), (summary or "").lower(), ticker.lower()

        # Ticker presence: 10 points in title, 5 points in summary
        score = (10.0 if ticker_lower in title_lower else 0) + \
                (5.0 if ticker_lower in summary_lower else 0)

        # Financial keywords: count matches (not normalized)
        keyword_matches = sum(1.0 if kw in title_lower else 0.5 if kw in summary_lower else 0
                             for kw in self.FINANCIAL_KEYWORDS)
        score += keyword_matches

        # Normalize to 0-1: base 15 (ticker match) + top 20% of keywords
        max_score = 15.0 + len(self.FINANCIAL_KEYWORDS) * 0.2
        return min(1.0, score / max_score) if max_score > 0 else 0.0

    def _calculate_article_score(self, title: str, summary: str, ticker: str) -> float:
        """Calculate score for ordering (no filtering)"""
        return self._calculate_relevance_score(title, summary, ticker)

    def _fetch_api(self, method: str, url: str, **kwargs) -> Optional[dict]:
        """Unified API request handler"""
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            return response.json() if response.status_code == 200 else None
        except Exception:
            return None

    def get_news_for_ticker(self, ticker: str, max_results: int = 20) -> List[NewsArticle]:
        """Fetch news from multiple sources, ordered by relevance score"""
        all_articles = []
        target = max_results * 2

        sources = [
            (self.fmp_key, self._get_fmp_articles),
            (self.tavily_key, self._get_tavily_articles),
            (self.finnhub_key, self._get_finnhub_articles),
            (self.alphavantage_key, self._get_alphavantage_articles)
        ]

        for key, fetch_func in sources:
            if key and len(all_articles) < target:
                all_articles.extend(fetch_func(ticker))

        # Score all articles (no filtering)
        for article in all_articles:
            article.relevance_score = self._calculate_article_score(article.title, article.summary, ticker)

        # Sort by score and return top results
        all_articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_articles[:max_results]

    def _parse_date(self, date_str: str, fmt: str = 'iso') -> Optional[datetime]:
        """Parse date: iso, timestamp, or compact format"""
        if not date_str:
            return None
        try:
            if fmt == 'iso':
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            elif fmt == 'timestamp':
                return datetime.fromtimestamp(date_str)
            elif fmt == 'compact':
                return datetime.strptime(date_str, '%Y%m%dT%H%M%S')
        except:
            return None

    def _get_fmp_articles(self, ticker: str) -> List[NewsArticle]:
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker.upper()}&limit=50&apikey={self.fmp_key}"
        data = self._fetch_api('GET', url)
        if not data:
            return []

        articles = []
        for item in data:
            publish_date = self._parse_date(item.get('publishedDate'))
            if publish_date and (datetime.now(publish_date.tzinfo) - publish_date).days > self.days_back:
                continue

            articles.append(NewsArticle(
                title=item.get('title', ''),
                url=item.get('url', ''),
                summary=item.get('text', '')[:500],
                publish_date=publish_date,
                source=item.get('site', 'FMP')
            ))
        return articles

    def _get_tavily_articles(self, query: str) -> List[NewsArticle]:
        data = self._fetch_api('POST', "https://api.tavily.com/search", json={
            "api_key": self.tavily_key,
            "query": f"{query} financial news stock market earnings",
            "search_depth": "advanced",
            "include_answer": False,
            "include_domains": self.TRUSTED_DOMAINS,
            "max_results": 20
        })

        return [NewsArticle(
            title=r.get('title', ''),
            url=r.get('url', ''),
            summary=r.get('content', '')[:500],
            publish_date=datetime.now(),
            source='Tavily'
        ) for r in data.get('results', [])] if data else []

    def _get_alphavantage_articles(self, ticker: str) -> List[NewsArticle]:
        data = self._fetch_api('GET', "https://www.alphavantage.co/query", params={
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker.upper(),
            'apikey': self.alphavantage_key,
            'limit': 50
        })
        if not data:
            return []

        articles = []
        for item in data.get('feed', []):
            publish_date = self._parse_date(item.get('time_published'), 'compact')
            if publish_date and (datetime.now() - publish_date).days > self.days_back:
                continue

            articles.append(NewsArticle(
                title=item.get('title', ''),
                url=item.get('url', ''),
                summary=item.get('summary', '')[:500],
                publish_date=publish_date,
                source=item.get('source', 'AlphaVantage')
            ))
        return articles

    def _get_finnhub_articles(self, ticker: str) -> List[NewsArticle]:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self.days_back)

        data = self._fetch_api('GET', "https://finnhub.io/api/v1/company-news", params={
            'symbol': ticker.upper(),
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        })

        return [NewsArticle(
            title=item.get('headline', ''),
            url=item.get('url', ''),
            summary=item.get('summary', '')[:500],
            publish_date=self._parse_date(item.get('datetime'), 'timestamp'),
            source=item.get('source', 'Finnhub')
        ) for item in data] if data else []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_news_for_ticker(ticker: str, max_results: int = 20, days_back: int = 7) -> List[NewsArticle]:
    """Get news articles for ticker, sorted by relevance"""
    retriever = NewsRetriever(days_back=days_back)
    return retriever.get_news_for_ticker(ticker, max_results=max_results)


def get_financial_news_for_analysis(ticker: str, max_results: int = 15, days_back: int = 7):
    """Get news articles + DataFrame + summary text for analysis"""
    articles = get_news_for_ticker(ticker, max_results=max_results, days_back=days_back)

    if not articles:
        return [], pd.DataFrame(), f"No news found for {ticker} in the last {days_back} days"

    df = pd.DataFrame([{
        'title': a.title,
        'source': a.source,
        'publish_date': a.publish_date,
        'summary_length': len(a.summary or ''),
        'title_length': len(a.title),
        'summary': a.summary,
        'url': a.url
    } for a in articles])

    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['date_only'] = df['publish_date'].dt.date
    df['hour'] = df['publish_date'].dt.hour

    return articles, df, f"Found {len(articles)} articles for {ticker} in the last {days_back} days"


def display_news_summary(articles: List[NewsArticle], df: pd.DataFrame, ticker: str):
    """Display formatted news summary with top articles and analytics"""
    if not articles:
        print("No articles to display")
        return

    print(f"\n📰 NEWS SUMMARY FOR {ticker}\n{'='*60}")
    print(f"\n📈 LATEST {min(5, len(articles))} ARTICLES:\n{'-'*50}")

    for i, article in enumerate(articles, 1):
        date_str = article.publish_date.strftime('%Y-%m-%d %H:%M') if article.publish_date else 'Recent'
        print(f"\n{i}. 📰 {article.title}")
        print(f"   📅 {date_str} | 📺 {article.source}")
        print(f"   📝 {(article.summary or 'No summary')[:150]}...")
        print(f"   🔗 {article.url}")

    if not df.empty:
        print(f"\n📊 NEWS ANALYTICS:\n{'-'*30}")
        print(f"• Total articles: {len(df)}")
        print(f"• Unique sources: {df['source'].nunique()}")
        print(f"• Date range: {df['publish_date'].min().strftime('%Y-%m-%d')} to {df['publish_date'].max().strftime('%Y-%m-%d')}")


def visualize_news_data(articles: List[NewsArticle], ticker: str, save_plots: bool = False):
    """Create 4-panel visualization: sources, title lengths, timeline, scatter"""
    import matplotlib.pyplot as plt

    if not articles:
        print("No articles to visualize")
        return

    df = pd.DataFrame([{
        'title': a.title,
        'source': a.source,
        'publish_date': a.publish_date,
        'summary_length': len(a.summary or ''),
        'title_length': len(a.title)
    } for a in articles])

    df['publish_date'] = pd.to_datetime(df['publish_date'], errors='coerce')
    df['date_only'] = df['publish_date'].dt.date
    df['hour'] = df['publish_date'].dt.hour

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Financial News Analysis for {ticker}', fontsize=16, fontweight='bold')

    # Panel 1: Source distribution
    source_counts = df['source'].value_counts().head(10)
    bars = axes[0, 0].bar(range(len(source_counts)), source_counts.values, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('News Sources Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('News Sources')
    axes[0, 0].set_ylabel('Number of Articles')
    axes[0, 0].set_xticks(range(len(source_counts)))
    axes[0, 0].set_xticklabels(source_counts.index, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    for bar, value in zip(bars, source_counts.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(value), ha='center', va='bottom')

    # Panel 2: Title length distribution
    axes[0, 1].hist(df['title_length'], bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Title Length Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Title Length (characters)')
    axes[0, 1].set_ylabel('Number of Articles')
    axes[0, 1].axvline(df['title_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["title_length"].mean():.0f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Publication timeline
    daily_counts = df['date_only'].value_counts().sort_index()
    axes[1, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2, markersize=8)
    axes[1, 0].set_title('Articles Published Over Time', fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Articles')
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

    # Panel 4: Source scatter (article count vs avg title length)
    source_stats = df.groupby('source').agg({'title_length': ['count', 'mean']}).round(1)
    source_stats.columns = ['article_count', 'avg_title_length']
    source_stats = source_stats[source_stats['article_count'] >= 2]
    if not source_stats.empty:
        axes[1, 1].scatter(source_stats['article_count'], source_stats['avg_title_length'],
                          s=source_stats['article_count']*50, alpha=0.6, color='green')
        axes[1, 1].set_title('Source Article Count vs Avg Title Length', fontweight='bold')
        axes[1, 1].set_xlabel('Article Count')
        axes[1, 1].set_ylabel('Average Title Length')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plots:
        filename = f"news_analysis_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {filename}")
    plt.show()

    print(f"\n📊 VISUALIZATION SUMMARY FOR {ticker}:\n{'-'*50}")
    print(f"• Total articles: {len(df)}")
    print(f"• Unique sources: {df['source'].nunique()}")
    print(f"• Title length range: {df['title_length'].min()}-{df['title_length'].max()} chars")
    print(f"• Most active: {df['source'].value_counts().index[0]} ({df['source'].value_counts().iloc[0]} articles)")


if __name__ == "__main__":
    print("🧪 Testing News Retriever\n" + "="*50)
    articles = get_news_for_ticker("AAPL", max_results=5, days_back=7)
    if articles:
        print(f"\n✅ Retrieved {len(articles)} articles")
    else:
        print("❌ No articles - check .env API keys")
