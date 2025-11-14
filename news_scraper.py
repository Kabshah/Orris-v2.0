from utils import *
from typing import Dict, List
import os
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
import asyncio

load_dotenv()


class NewsScraper:
    _rate_limiter = AsyncLimiter(5, 1)

    async def scrape_news(self, topics: List[str]) -> Dict[str, str]:
        """Scrape and analyze news articles"""
        results = {}

        for topic in topics:
            async with self._rate_limiter:
                try:
                    urls = generate_news_urls_to_scrape([topic])
                    search_html = scrape_with_brightdata(urls[topic])
                    clean_text = clean_html_to_text(search_html)
                    headlines = extract_headlines(clean_text)
                    
                    # Using HF API instead of Groq
                    summary = summarize_with_hf_news_script(
                        hf_api_key=os.getenv("HF_API_KEY"),
                        headlines=headlines
                    )
                    results[topic] = summary
                except Exception as e:
                    results[topic] = f"Error: {str(e)}"

                await asyncio.sleep(1)
        
        return {"news_analysis": results}
