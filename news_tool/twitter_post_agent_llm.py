#!/usr/bin/env python3
"""
Twitter Post Generation Agent - LLM-Powered Financial News to Engaging Tweets

Uses Google Gemini to transform evaluated news articles into compact, engaging,
funny yet informative Twitter posts about financial topics.
"""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import google.generativeai as genai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from textwrap import wrap
import warnings

# Suppress emoji font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Suppress gRPC ALTS warnings (not running on GCP)
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

load_dotenv()

@dataclass
class TwitterPost:
    """Generated Twitter post with metadata"""
    tweet_text: str
    character_count: int
    news_source: str
    news_title: str
    news_url: str
    news_rank: int  # Rank from evaluated news (1 = most important)
    hashtags: List[str]
    tone: str  # e.g., "witty", "analytical", "excited", "cautious"
    hook_type: str  # e.g., "stat", "question", "hot take", "meme"


class TwitterPostAgent:
    """LLM-powered Twitter post generator using Google Gemini"""

    DEFAULT_STYLE_GUIDE = """
    Create informative, investment-focused Twitter posts about financial news that are:

    **Voice & Tone:**
    - Professional but conversational
    - Fact-based and analytical
    - Calm confidence - no hype or urgency
    - Let the data speak
    - Smart investor talking to smart investors

    **Length Requirements:**
    - Target: 180-240 characters
    - 2-3 sentences maximum
    - Concise but complete thoughts
    - Use line breaks for readability

    **Content Structure:**
    Opening statement (investment thesis or observation)

    Supporting facts (specific numbers, metrics, trends)

    Hashtags (2 max)

    **What to INCLUDE:**
    - Specific numbers and percentages
    - Key financial metrics (revenue growth, margins, ROIC, EPS, etc.)
    - Timeframes for context ("since 2021", "Q3 results", "YoY")
    - Clear investment implications
    - Ticker symbol naturally integrated

    **What to AVOID:**
    - Sensational language ("shocking", "breaking", "explosive")
    - Questions that feel clickbait-y
    - Emojis overuse (max 1, or none)
    - Hype or fear-mongering
    - Explicit buy/sell advice

    **Formatting:**
    - Use line breaks (\n\n) between sentences for clarity
    - Lead with ticker symbol when natural
    - Numbers should stand out
    - Keep hashtags minimal and relevant

    **Example of GOOD investment-focused tweet:**
    $ELF is an incredible bargain. Steady revenue growth and 3x increase in ROIC and margins since 2021.

    #Stocks #ELF

    **Another good example:**
    $NVDA data center revenue up 112% YoY to $30.8B.

    AI demand isn't slowing down. Margins holding at 75%.

    #Nvidia #AI

    **Example of BAD (too sensational):**
    🚨 BREAKING: $AAPL crashes 6.8% in Brazil! 📉

    Competition heating up FAST!

    Is this the END for Apple? 😱

    #Stocks #AAPL
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize the Twitter post generation agent"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        print(f"✅ Twitter Post Agent initialized with {model_name}")

    def generate_posts(
        self,
        evaluated_news: List[Any],
        ticker: str,
        company_name: Optional[str] = None,
        posts_per_article: int = 1,
        custom_instructions: Optional[str] = None
    ) -> List[TwitterPost]:
        """
        Generate Twitter posts from evaluated news articles

        Args:
            evaluated_news: List of EvaluatedNews objects from news_evaluation_agent
            ticker: Stock ticker symbol
            company_name: Optional company name
            posts_per_article: Number of different posts to generate per article (default: 1)
            custom_instructions: Optional additional guidelines for tweet generation

        Returns:
            List of TwitterPost objects with different angles/tones
        """
        if not evaluated_news:
            print("❌ No evaluated news to generate posts from")
            return []

        print(f"\n🐦 Generating Twitter posts for {ticker}...")

        all_posts = []

        for i, news in enumerate(evaluated_news, 1):
            print(f"  📝 Generating post {i}/{len(evaluated_news)} (Rank #{news.rank})...")

            # Prepare news data for prompt
            news_data = {
                'rank': news.rank,
                'title': news.title,
                'summary': news.summary[:500],
                'source': news.source,
                'url': news.url,
                'evaluation_reason': news.evaluation_reason,
                'key_insights': news.key_insights
            }

            # Build generation prompt
            style_guide = self.DEFAULT_STYLE_GUIDE
            company_context = f" ({company_name})" if company_name else ""

            # Add custom instructions if provided
            custom_section = ""
            if custom_instructions:
                custom_section = f"""
CUSTOM INSTRUCTIONS (MUST FOLLOW):
{custom_instructions}
"""

            prompt = f"""You are a creative financial content writer crafting Twitter posts about ${ticker}{company_context}.

STYLE GUIDE:
{style_guide}
{custom_section}
NEWS ARTICLE TO TRANSFORM (Rank #{news_data['rank']} from evaluated news):
Title: {news_data['title']}
Summary: {news_data['summary']}
Why It Matters: {news_data['evaluation_reason']}
Key Insights:
{chr(10).join(f'- {insight}' for insight in news_data['key_insights'])}

SOURCE: {news_data['source']}
URL: {news_data['url']}

NOTE: This is ranked #{news_data['rank']} in importance among the evaluated news.

TASK:
Generate {posts_per_article} different Twitter post(s) about this news. Each post should have a different angle or tone.

For each post, provide:
1. **tweet_text** - The complete tweet (max 240 characters, target 180-220)
2. **hashtags** - 2 relevant hashtags (included in character count)
3. **tone** - One word describing the tone (e.g., "analytical", "confident", "observational", "factual")
4. **hook_type** - Type of hook used (e.g., "thesis", "metric", "comparison", "trend")

Return your response as a JSON array with this structure:
```json
[
  {{
    "tweet_text": "$TICKER investment thesis or observation.\\n\\nSpecific metrics and numbers with context.\\n\\n#Stocks #TICKER",
    "hashtags": ["Stocks", "TICKER"],
    "tone": "analytical",
    "hook_type": "metric"
  }}
]
```

CRITICAL REQUIREMENTS:
- Target: 180-240 characters total
- Use \\n\\n (double backslash n) for line breaks in JSON - this is REQUIRED for valid JSON
- 2-3 sentences maximum (not 4+ lines)
- Include ticker symbol ${ticker} naturally in the text
- Focus on financial metrics and data points
- Professional tone - NO sensational language
- Let the numbers tell the story

CONTENT GUIDELINES:
- Lead with investment thesis or key observation
- Include specific metrics (revenue growth %, margin expansion, ROIC, EPS, etc.)
- Add timeframe context (YoY, QoQ, since 2021, Q3, etc.)
- No emojis or minimal use (0-1 max)
- No clickbait questions or hype

STRUCTURE FORMAT:
Sentence 1: Investment observation or thesis\\n\\nSentence 2: Supporting metrics with specific numbers and timeframes\\n\\n#Hashtag1 #Hashtag2

IMPORTANT: In the JSON, you MUST write \\n\\n (with double backslash) for newlines, NOT actual line breaks.
The tweet_text field must be a single line in JSON with \\n\\n escape sequences.

EXAMPLE STYLE TO MATCH:
"$ELF is an incredible bargain. Steady revenue growth and 3x increase in ROIC and margins since 2021.\\n\\n#Stocks #ELF"

Respond ONLY with the JSON array, no additional text."""

            try:
                # Call Gemini API
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()

                # Extract JSON from response
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif '```' in response_text:
                    json_start = response_text.find('```') + 3
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()

                # Parse generated posts
                generated_posts = json.loads(response_text)

                # Create TwitterPost objects
                for post_data in generated_posts:
                    tweet_text = post_data['tweet_text']
                    char_count = len(tweet_text)

                    if char_count > 280:
                        print(f"    ⚠️  Tweet too long ({char_count} chars), skipping...")
                        continue

                    all_posts.append(TwitterPost(
                        tweet_text=tweet_text,
                        character_count=char_count,
                        news_source=news.source,
                        news_title=news.title,
                        news_url=news.url,
                        news_rank=news.rank,
                        hashtags=post_data.get('hashtags', []),
                        tone=post_data.get('tone', 'unknown'),
                        hook_type=post_data.get('hook_type', 'unknown')
                    ))

            except json.JSONDecodeError as e:
                print(f"    ❌ Failed to parse response: {e}")
                continue
            except Exception as e:
                print(f"    ❌ Generation failed: {e}")
                continue

        print(f"  ✅ Generated {len(all_posts)} Twitter posts")
        return all_posts

    def display_posts(self, posts: List[TwitterPost], ticker: str):
        """Display formatted Twitter posts"""
        if not posts:
            print("No posts to display")
            return

        print(f"\n{'='*80}")
        print(f"🐦 GENERATED TWITTER POSTS FOR ${ticker}")
        print(f"{'='*80}")

        for i, post in enumerate(posts, 1):
            print(f"\n📱 POST #{i} (Based on News Rank #{post.news_rank})")
            print(f"{'─'*80}")
            print(f"Tone: {post.tone.title()} | Hook: {post.hook_type.title()} | Length: {post.character_count}/280 chars")
            print(f"\n{post.tweet_text}")
            print(f"\n📰 Source Article: {post.news_source}")
            print(f"📌 Title: {post.news_title[:70]}...")
            print(f"🔗 URL: {post.news_url}")
            print(f"{'─'*80}")

    def save_posts(self, posts: List[TwitterPost], ticker: str, output_file: Optional[str] = None) -> str:
        """Save generated posts to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"twitter_posts_{ticker}_{timestamp}.json"

        results = {
            'ticker': ticker,
            'generation_date': datetime.now().isoformat(),
            'total_posts': len(posts),
            'posts': [asdict(post) for post in posts]
        }

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Posts saved: {output_file}")
        return output_file

    def save_as_images(
        self,
        posts: List[TwitterPost],
        ticker: str,
        username: str = "YourHandle",
        output_dir: Optional[str] = None
    ) -> List[str]:
        """Save Twitter posts as realistic tweet images

        Args:
            posts: List of TwitterPost objects
            ticker: Stock ticker symbol
            username: Twitter handle to display
            output_dir: Directory to save images (default: current directory)

        Returns:
            List of saved image file paths
        """
        if not posts:
            print("No posts to save as images")
            return []

        if output_dir is None:
            output_dir = "."
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []

        # Configure matplotlib to suppress emoji warnings
        plt.rcParams['font.family'] = 'DejaVu Sans'

        for i, post in enumerate(posts, 1):
            # Create compact figure
            fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='#f7f9f9')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.axis('off')

            # Twitter/X card
            card = mpatches.FancyBboxPatch(
                (4, 8), 92, 84,
                boxstyle="round,pad=1.2",
                edgecolor='#cfd9de',
                facecolor='white',
                linewidth=1.5
            )
            ax.add_patch(card)

            # ===== HEADER SECTION =====
            # Profile circle
            profile_circle = plt.Circle((9, 84), 2.5, color='#1d9bf0', alpha=0.2)
            ax.add_patch(profile_circle)
            ax.text(9, 84, username[0].upper(), fontsize=11, weight='bold',
                   color='#1d9bf0', ha='center', va='center')

            # Username and handle
            ax.text(13.5, 85.5, username, fontsize=10, weight='bold', color='#0f1419')
            ax.text(13.5, 82.5, f'@{username}', fontsize=8, color='#536471')

            # Timestamp
            ax.text(92, 84, 'now', fontsize=8, color='#536471', ha='right')

            # Header separator
            ax.plot([7, 93], [79, 79], color='#eff3f4', linewidth=1.2)

            # ===== TWEET TEXT SECTION =====
            # Handle line breaks for spacing (tweets now use \n\n for spacing)
            text_lines = post.tweet_text.split('\n')

            y_pos = 74
            base_line_height = 4.5

            for line in text_lines:
                if line.strip():  # Non-empty line
                    # Escape $ signs for matplotlib (LaTeX interpreter treats $ as math mode)
                    line_escaped = line.replace('$', r'\$')

                    # Wrap long lines if needed
                    if len(line) > 60:
                        wrapped = wrap(line, width=60)
                        for wrapped_line in wrapped:
                            wrapped_escaped = wrapped_line.replace('$', r'\$')
                            ax.text(9, y_pos, wrapped_escaped, fontsize=10.5, color='#0f1419',
                                   va='top', wrap=False)
                            y_pos -= base_line_height
                    else:
                        ax.text(9, y_pos, line_escaped, fontsize=10.5, color='#0f1419',
                               va='top', wrap=False)
                        y_pos -= base_line_height
                else:  # Empty line - add extra spacing
                    y_pos -= 2.5

            # Extra spacing after tweet text
            y_pos -= 2

            # ===== URL SECTION (NEW) =====
            # Shorten URL for display
            url_display = post.news_url
            if len(url_display) > 50:
                url_display = url_display[:47] + '...'

            # URL box with light background
            url_box = mpatches.FancyBboxPatch(
                (8, y_pos - 8), 84, 6,
                boxstyle="round,pad=0.3",
                edgecolor='#cfd9de',
                facecolor='#f7f9f9',
                linewidth=0.8
            )
            ax.add_patch(url_box)

            # URL text (clickable-looking) - escape $ if present
            url_display_escaped = url_display.replace('$', r'\$')
            ax.text(9.5, y_pos - 5, '🔗 ' + url_display_escaped, fontsize=7.5,
                   color='#1d9bf0', style='italic')

            y_pos -= 11

            # ===== ENGAGEMENT BAR =====
            ax.plot([7, 93], [y_pos, y_pos], color='#eff3f4', linewidth=1.2)

            engagement_y = y_pos - 5
            button_spacing = 20
            ax.text(12, engagement_y, 'Reply', fontsize=8.5, color='#536471')
            ax.text(32, engagement_y, 'Retweet', fontsize=8.5, color='#536471')
            ax.text(55, engagement_y, 'Like', fontsize=8.5, color='#536471')
            ax.text(75, engagement_y, 'Share', fontsize=8.5, color='#536471')

            # ===== METADATA FOOTER =====
            footer_y = engagement_y - 7
            ax.plot([7, 93], [footer_y + 2, footer_y + 2], color='#eff3f4', linewidth=1.2)

            # Source info - split into two lines for readability (escape $ if present)
            source_escaped = post.news_source.replace('$', r'\$')
            ax.text(9, footer_y - 1.5, f"News Rank #{post.news_rank}",
                   fontsize=7.5, color='#536471', weight='bold')
            ax.text(9, footer_y - 4.5, f"Source: {source_escaped}",
                   fontsize=7.5, color='#536471', style='italic')

            # Character count
            ax.text(92, footer_y - 1.5, f"{post.character_count}/280",
                   fontsize=7.5, color='#536471', ha='right')

            # Tone and hook
            ax.text(92, footer_y - 4.5, f"{post.tone} • {post.hook_type}",
                   fontsize=7.5, color='#1d9bf0', ha='right')

            # Save with high quality
            filename = f"twitter_{ticker}_{timestamp}_post{i}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=200, bbox_inches='tight',
                       facecolor='#f7f9f9', edgecolor='none',
                       pad_inches=0.2)
            plt.close()

            saved_files.append(filepath)
            print(f"  💾 Saved: {filename}")

        print(f"\n✅ Saved {len(saved_files)} tweet images to {output_dir}/")
        return saved_files


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_twitter_posts_llm(
    evaluated_news: List[Any],
    ticker: str,
    company_name: Optional[str] = None,
    posts_per_article: int = 1,
    custom_instructions: Optional[str] = None,
    display_posts: bool = True,
    save_json: bool = False,
    save_image: bool = False,
    image_dir: Optional[str] = None,
    username: str = "YourHandle"
) -> List[TwitterPost]:
    """
    Generate Twitter posts from evaluated news (simple, news-only approach)

    Args:
        evaluated_news: List of EvaluatedNews from news_evaluation_agent
        ticker: Stock ticker symbol
        company_name: Optional company name
        posts_per_article: Number of posts per article (default: 1)
        custom_instructions: Optional additional guidelines for tweet generation
        display_posts: Whether to print formatted posts (default: True)
        save_json: Whether to save to JSON file (default: False)
        save_image: Whether to save as tweet images (default: False)
        image_dir: Directory to save images (default: './tweet_images')
        username: Twitter handle for image (default: 'YourHandle')

    Returns:
        List of TwitterPost objects
    """
    agent = TwitterPostAgent()

    posts = agent.generate_posts(
        evaluated_news=evaluated_news,
        ticker=ticker,
        company_name=company_name,
        posts_per_article=posts_per_article,
        custom_instructions=custom_instructions
    )

    if display_posts and posts:
        agent.display_posts(posts, ticker)

    if save_json and posts:
        agent.save_posts(posts, ticker)

    if save_image and posts:
        output_dir = image_dir if image_dir else './tweet_images'
        agent.save_as_images(posts, ticker, username, output_dir)

    return posts


if __name__ == "__main__":
    from financial_news import get_news_for_ticker
    from news_evaluation_agent import evaluate_news_articles

    print("🧪 Testing Twitter Post Generator")
    ticker = "AAPL"
    articles = get_news_for_ticker(ticker, max_results=30, days_back=1)

    if articles:
        top_news = evaluate_news_articles(
            articles=articles,
            ticker=ticker,
            company_name="Apple Inc.",
            top_k=3,
            display_results=False
        )
        posts = generate_twitter_posts_llm(
            evaluated_news=top_news,
            ticker=ticker,
            company_name="Apple Inc.",
            posts_per_article=2,
            display_posts=True,
            save_json=True
        )
        if posts:
            print(f"\n✅ Generated {len(posts)} posts")
    else:
        print("❌ No articles retrieved")
