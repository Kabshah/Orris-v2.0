from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from models import NewsRequest
from reddit_scraper import scrape_reddit_topics
from utils import *
import os
import traceback

app = FastAPI()
load_dotenv()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from news_scraper import NewsScraper


@app.post("/generate-news-audio")
async def generate_news_audio(request: NewsRequest):
    try:
        results = {}
        
        # Scrape News
        if request.source_type in ["news", "both"]:
            try:
                print(f"[INFO] Scraping news for topics: {request.topics}")
                news_scraper = NewsScraper()
                results["news"] = await news_scraper.scrape_news(request.topics)
                print(f"[INFO] News scraping completed")
            except Exception as e:
                print(f"[ERROR] News scraping failed: {str(e)}")
                results["news"] = {"news_analysis": {topic: f"News unavailable for {topic}" for topic in request.topics}}

        # Scrape Reddit
        if request.source_type in ["reddit", "both"]:
            try:
                print(f"[INFO] Scraping Reddit for topics: {request.topics}")
                results["reddit"] = await scrape_reddit_topics(request.topics)
                print(f"[INFO] Reddit scraping completed")
            except Exception as e:
                print(f"[ERROR] Reddit scraping failed: {str(e)}")
                traceback.print_exc()
                results["reddit"] = {"reddit_analysis": {topic: f"Reddit analysis unavailable" for topic in request.topics}}

        news_data = results.get("news", {})
        reddit_data = results.get("reddit", {})
        
        print(f"[INFO] Generating broadcast news...")
        
        # Generate broadcast news using HF API
        try:
            news_summary = generate_brodcast_news(
                hf_api_key=os.getenv("HF_API_KEY"),
                news_data=news_data,
                reddit_data=reddit_data,
                topics=request.topics
            )
            print(f"[INFO] Broadcast news generated successfully")
        except Exception as e:
            print(f"[ERROR] Broadcast generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate news summary: {str(e)}")
        
        print(f"[INFO] Converting text to audio...")
        
        # Convert summary to audio
        try:
            audio_path = text_to_audio_elevenlabs_sdk(
                text=news_summary,
                voice_id="iPmVSMDLX3FRQaONHDW2",
                model_id="eleven_multilingual_v2",
                output_dir="audio"
            )
            print(f"[INFO] Audio generated at: {audio_path}")
        except Exception as e:
            print(f"[ERROR] Audio generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")
        
        if audio_path:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            print(f"[INFO] Returning audio response")
            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={"Content-Disposition": "attachment; filename=news-summary.mp3"}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unhandled error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


import uvicorn
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=1234, reload=True)
