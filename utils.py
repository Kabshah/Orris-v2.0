# from urllib.parse import quote_plus
# import os
# import requests
# from fastapi import FastAPI,HTTPException
# from bs4 import BeautifulSoup
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_groq import ChatGroq
# from elevenlabs import ElevenLabs
# from datetime import datetime

# from dotenv import load_dotenv
# load_dotenv()

# def generate_valid_news_url(keyword:str) -> str:
#     """
#     Generate A Google News search URL for a keyword
#     Args:
#        Keyword :Search term to use in the news search
#     Returns:
#        str: Constructed google news search Url
#     """

#     q = quote_plus(keyword)
#     return f"https://news.google.com/search?q={q}&tbs=sbd:1"

# def scrape_with_brightdata(url) -> str:
#     ''' Scrap url using BrightData'''

#     headers = {
#         "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "zone": os.getenv("WEB_UNLOCKER_ZONE"),
#         "url":url,
#         "format":"raw"
#     }
#     try:
#         response = requests.post("https://api.brightdata.com/request",json=payload,headers=headers)
#         response.raise_for_status()
#         return response.text   #yeh html laakr dey rhA h
#     except requests.exceptions.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"BrightData error:{str(e)}")
    

# def clean_html_to_text(html_content:str) -> str:
#     """ Clean  HTML to plain text"""
#     soup = BeautifulSoup(html_content,"html.parser")
#     text= soup.get_text(separator="\n")
#     return text.strip()


# def extract_headlines(cleaned_text:str) -> str:
#     """
#     Extract and concatenate headlines from cleaned news text content.
#     Args:
#         cleaned_text: Raw text from HTML cleaning
#     Returns: 
#         str:Combined headlines seprated by newlines
#     """
#     headlines = []
#     current_block = []

#     lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]

#     for line in lines:
#         if line == "More":
#             # agr line ma additional information h toh append in same 
#             if current_block:
#                 # append in same line
#                 headlines.append(current_block[0])
#                 current_block = []
#             else:
#                 current_block.append(line)

#     # add remaining block at the end of the text
#     if current_block:
#         headlines.append(current_block[0])

#     return "\n".join(headlines)


# def summarize_with_groq_news_script(groq_api_key: str, headlines: str) -> str:
#     """
#     Summarize multiple news headlines into a TTS-friendly broadcast news script using Anthropic Claude model via LangChain.
#     """
#     system_prompt = """
# You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.

# The final output will be read aloud by a news anchor or text-to-speech engine. So:
# - Do not include any special characters, emojis, formatting symbols, or markdown.
# - Do not add any preamble or framing like "Here's your summary" or "Let me explain".
# - Write in full, clear, spoken-language paragraphs.
# - Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
# - Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
# - Start right away with the actual script, using transitions between topics if needed.

# Remember: Your only output should be a clean script that is ready to be read out loud.
# """

#     try:
#         llm = ChatGroq(
#             model="qwen/qwen3-32b",  
#             # groq_api_key=os.getenv("GROQ_API_KEY"),
#             groq_api_key=groq_api_key,
#             temperature=0.4,
#             max_tokens=1000
#         )

#         # Invoke Claude with system + user prompt
#         response = llm.invoke([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=headlines)
#         ])

#         return response.content
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Groq error: {str(e)}")
    

#   # value -> actual url
#  #  key -> keyword
# def generate_news_urls_to_scrape(list_of_keywords):
#     valid_urls_dict = {}
#     for keyword in list_of_keywords:
#         valid_urls_dict[keyword] = generate_valid_news_url(keyword)
#     return valid_urls_dict


# # def generate_brodcast_news(groq_api_key,news_data, reddit_data, topics):
# #     system_prompt = """
# #     You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

# #     For each topic, STRUCTURE BASED ON AVAILABLE DATA:
# #     1. If news exists: "According to official reports..." + summary
# #     2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
# #     3. If both exist: Present news first, then Reddit reactions
# #     4. If neither exists: Skip the topic (shouldn't happen)

# #     Formatting rules:
# #     - ALWAYS start directly with the content, NO INTRODUCTIONS
# #     - Keep audio length 60-120 seconds per topic
# #     - Use natural speech transitions like "Meanwhile, online discussions..." 
# #     - Incorporate 1-2 short quotes from Reddit when available
# #     - Maintain neutral tone but highlight key sentiments
# #     - End with "To wrap up this segment..." summary

# #     Write in full paragraphs optimized for speech synthesis. Avoid markdown.
# #     """

# #     try:
# #         # DEBUG: Check what data we're receiving
# #         print(f"DEBUG: Topics received: {topics}")
# #         print(f"DEBUG: News data type: {type(news_data)}")
# #         print(f"DEBUG: Reddit data type: {type(reddit_data)}")
# #         topic_blocks = []
# #         for topic in topics:
# #             # pehly ek topic ke news summary etc karo then then dosry topic prr joa
# #             # phly news discussion then reddit discussion
# #             news_content = news_data["news_analysis"].get(topic) if news_data else " "
# #             reddit_content = reddit_data["reddit_analysis"].get(topic) if reddit_data else " "
# #             context = []
# #             if news_content:
# #                 context.append(f"Official news content:\n{news_content}")

# #             if reddit_content:
# #                 context.append(f" Reddit Discusion content:\n{reddit_content}")

# #             if context:
# #                 topic_blocks.append(
# #                     f"TOPIC:{topic}\n\n"+
# #                     "\n\n".join(context)
# #                 )

# #         user_prompt = {
# #             "Create brodcast segments for these topics using available sources:\n\n"+
# #             "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)
# #         }

# #         llm = ChatGroq(
# #             model = "qwen/qwen3-32b",
# #             #groq_api_key=os.getenv("GROQ_API_KEY"),
# #             groq_api_key=groq_api_key,
# #             temperature=0.3,
# #             max_tokens=4000
# #         )
# #         response = llm.invoke([
# #             SystemMessage(content=system_prompt),
# #             HumanMessage(content=user_prompt)
# #         ])
# #         return response.content
# #     except Exception as e:
# #         raise e
    

# # def text_to_audio_elevenlabs_sdk(
# #         text:str,
# #         voice_id : str = "iPmVSMDLX3FRQaONHDW2",
# #         model_id : str = "eleven_multilingual_v2",
# #         output_format: str ="mp3_44100_128",
# #         output_dir: str = "audio",
# #         api_key: str = None
# # ) -> str:
# #     """
# #     Convert text to speech using ElevenLabs SDK and save it to audio.directory.
# #     Returns:
# #            str: Path to the saved audio file.
# #     """
# #     try:
# #         api_key = os.getenv("ELEVENLABS_API_KEY")
# #         if not api_key:
# #             raise ValueError("ElevenLabs API Key is required.")
        
# #         # Initialize Client
# #         client = ElevenLabs(api_key=api_key)

# #         #Get the audio generator
# #         audio_stream = client.text_to_speech.convert(
# #             text=text,
# #             voice_id=voice_id,
# #             model_id=model_id,
# #             output_format=output_format
# #         )

# #         # ensure the directory exists
# #         os.makedirs(output_dir, exist_ok=True)

# #         # generate unique file name (jistie prr genrate hori h wo daal do name ma)
# #         filename= f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')},mp3"
# #         filepath = os.path.join(output_dir, filename)

# #         # write audio chunks to file
# #         with open(filepath,"wb") as f:
# #             for chunk in audio_stream:
# #                 f.write(chunk)

# #             return filepath
        
# #     except Exception as e:
# #         raise e 

# def generate_brodcast_news(groq_api_key, news_data, reddit_data, topics):
#     system_prompt = """
#     You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

#     For each topic, STRUCTURE BASED ON AVAILABLE DATA:
#     1. If news exists: "According to official reports..." + summary
#     2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
#     3. If both exist: Present news first, then Reddit reactions
#     4. If neither exists: Skip the topic (shouldn't happen)

#     Formatting rules:
#     - ALWAYS start directly with the content, NO INTRODUCTIONS
#     - Keep audio length 60-120 seconds per topic
#     - Use natural speech transitions like "Meanwhile, online discussions..." 
#     - Incorporate 1-2 short quotes from Reddit when available
#     - Maintain neutral tone but highlight key sentiments
#     - End with "To wrap up this segment..." summary

#     Write in full paragraphs optimized for speech synthesis. Avoid markdown.
#     """

#     try:
#         print(f"DEBUG: Topics received: {topics}")
#         print(f"DEBUG: News data type: {type(news_data)}")
#         print(f"DEBUG: Reddit data type: {type(reddit_data)}")
        
#         topic_blocks = []
#         for topic in topics:
#             news_content = news_data.get("news_analysis", {}).get(topic, "") if news_data else ""
#             reddit_content = reddit_data.get("reddit_analysis", {}).get(topic, "") if reddit_data else ""
            
#             context = []
#             if news_content and news_content != "Error: ":
#                 context.append(f"Official news content:\n{news_content}")

#             if reddit_content and reddit_content != "Error: ":
#                 context.append(f"Reddit Discussion content:\n{reddit_content}")

#             if context:
#                 topic_blocks.append(
#                     f"TOPIC: {topic}\n\n" +
#                     "\n\n".join(context)
#                 )

#         user_prompt = "Create broadcast segments for these topics using available sources:\n\n" + "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)

#         llm = ChatGroq(
#             model="qwen/qwen3-32b",
#             groq_api_key=groq_api_key,
#             temperature=0.3,
#             max_tokens=4000
#         )
        
#         response = llm.invoke([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt)
#         ])
#         return response.content
#     except Exception as e:
#         raise e
    




# def text_to_audio_elevenlabs_sdk(
#         text:str,
#         voice_id : str = "iPmVSMDLX3FRQaONHDW2",
#         model_id : str = "eleven_multilingual_v2",
#         output_format: str ="mp3_44100_128",
#         output_dir: str = "audio",
#         api_key: str = None
# ) -> str:
#     """
#     Convert text to speech using ElevenLabs SDK and save it to audio.directory.
#     Returns:
#            str: Path to the saved audio file.
#     """
#     try:
#         api_key = os.getenv("ELEVENLABS_API_KEY")
#         if not api_key:
#             raise ValueError("ElevenLabs API Key is required.")
        
#         # Initialize Client
#         client = ElevenLabs(api_key=api_key)

#         #Get the audio generator
#         audio_stream = client.text_to_speech.convert(
#             text=text,
#             voice_id=voice_id,
#             model_id=model_id,
#             output_format=output_format
#         )

#         # ensure the directory exists
#         os.makedirs(output_dir, exist_ok=True)

#         # generate unique file name (jistie prr genrate hori h wo daal do name ma)
#         filename= f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')},mp3"
#         filepath = os.path.join(output_dir, filename)

#         # write audio chunks to file
#         with open(filepath,"wb") as f:
#             for chunk in audio_stream:
#                 f.write(chunk)

#             return filepath
        
#     except Exception as e:
#         raise e 










from urllib.parse import quote_plus
import os
import requests
from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup
from elevenlabs import ElevenLabs
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

def generate_valid_news_url(keyword: str) -> str:
    """Generate A Google News search URL for a keyword"""
    q = quote_plus(keyword)
    return f"https://news.google.com/search?q={q}&tbs=sbd:1"


def scrape_with_brightdata(url) -> str:
    """Scrape url using BrightData"""
    headers = {
        "Authorization": f"Bearer {os.getenv('BRIGHTDATA_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "zone": os.getenv("WEB_UNLOCKER_ZONE"),
        "url": url,
        "format": "raw"
    }
    try:
        response = requests.post("https://api.brightdata.com/request", json=payload, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"BrightData error:{str(e)}")


def clean_html_to_text(html_content: str) -> str:
    """Clean HTML to plain text"""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text.strip()


def extract_headlines(cleaned_text: str) -> str:
    """Extract and concatenate headlines from cleaned news text content"""
    headlines = []
    current_block = []

    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]

    for line in lines:
        if line == "More":
            if current_block:
                headlines.append(current_block[0])
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        headlines.append(current_block[0])

    return "\n".join(headlines)


def summarize_with_hf_news_script(hf_api_key: str, headlines: str) -> str:
    """
    Summarize multiple news headlines into a TTS-friendly broadcast news script 
    using Hugging Face Inference API with Llama model.
    """
    system_prompt = """You are my personal news editor and scriptwriter for a news podcast. Your job is to turn raw headlines into a clean, professional, and TTS-friendly news script.

The final output will be read aloud by a news anchor or text-to-speech engine. So:
- Do not include any special characters, emojis, formatting symbols, or markdown.
- Do not add any preamble or framing like "Here's your summary" or "Let me explain".
- Write in full, clear, spoken-language paragraphs.
- Keep the tone formal, professional, and broadcast-style — just like a real TV news script.
- Focus on the most important headlines and turn them into short, informative news segments that sound natural when spoken.
- Start right away with the actual script, using transitions between topics if needed.

Remember: Your only output should be a clean script that is ready to be read out loud."""

    try:
        client = InferenceClient(api_key=hf_api_key)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": headlines}
        ]
        
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            temperature=0.4,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF Inference error: {str(e)}")


def generate_news_urls_to_scrape(list_of_keywords):
    """Generate Google News URLs for list of keywords"""
    valid_urls_dict = {}
    for keyword in list_of_keywords:
        valid_urls_dict[keyword] = generate_valid_news_url(keyword)
    return valid_urls_dict


def generate_brodcast_news(hf_api_key, news_data, reddit_data, topics):
    """Generate broadcast news using HF Inference API"""
    system_prompt = """You are broadcast_news_writer, a professional virtual news reporter. Generate natural, TTS-ready news reports using available sources:

For each topic, STRUCTURE BASED ON AVAILABLE DATA:
1. If news exists: "According to official reports..." + summary
2. If Reddit exists: "Online discussions on Reddit reveal..." + summary
3. If both exist: Present news first, then Reddit reactions
4. If neither exists: Skip the topic (shouldn't happen)

Formatting rules:
- ALWAYS start directly with the content, NO INTRODUCTIONS
- Keep audio length 60-120 seconds per topic
- Use natural speech transitions like "Meanwhile, online discussions..." 
- Incorporate 1-2 short quotes from Reddit when available
- Maintain neutral tone but highlight key sentiments
- End with "To wrap up this segment..." summary

Write in full paragraphs optimized for speech synthesis. Avoid markdown."""

    try:
        print(f"DEBUG: Topics received: {topics}")
        print(f"DEBUG: News data type: {type(news_data)}")
        print(f"DEBUG: Reddit data type: {type(reddit_data)}")
        
        topic_blocks = []
        for topic in topics:
            news_content = news_data.get("news_analysis", {}).get(topic, "") if news_data else ""
            reddit_content = reddit_data.get("reddit_analysis", {}).get(topic, "") if reddit_data else ""
            
            context = []
            if news_content and news_content != "Error: ":
                context.append(f"Official news content:\n{news_content}")

            if reddit_content and reddit_content != "Error: ":
                context.append(f"Reddit Discussion content:\n{reddit_content}")

            if context:
                topic_blocks.append(
                    f"TOPIC: {topic}\n\n" +
                    "\n\n".join(context)
                )

        user_prompt = "Create broadcast segments for these topics using available sources:\n\n" + "\n\n--- NEW TOPIC ---\n\n".join(topic_blocks)

        client = InferenceClient(api_key=hf_api_key)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=4000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        raise e


def text_to_audio_elevenlabs_sdk(
        text: str,
        voice_id: str = "iPmVSMDLX3FRQaONHDW2",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
        output_dir: str = "audio",
        api_key: str = None
) -> str:
    """Convert text to speech using ElevenLabs SDK"""
    try:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API Key is required.")
        
        client = ElevenLabs(api_key=api_key)

        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )

        os.makedirs(output_dir, exist_ok=True)

        filename = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return filepath
        
    except Exception as e:
        raise e