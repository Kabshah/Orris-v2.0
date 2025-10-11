# from typing import List
# import os
# import asyncio
# from utils import *
# from datetime import datetime,timedelta
# from dotenv import load_dotenv
# from aiolimiter import AsyncLimiter
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_groq import ChatGroq
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent

# load_dotenv()

# #setting up server parameters of mcp
# server_params= StdioServerParameters(
#     command="npx",
#     env={
#         "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN"),
#         "RATE_LIMIT": "100/1h",          # Custom rate limiting
#         "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),   # Custom unlocker zone
#         #"BROWSER_ZONE": "custom_browser" # Custom browser zone
#         #"PRO_MODE": "true",              # Enable all 60+ tools
#       },
#       args=["@brightdata/mcp"]
#     )
# model = ChatGroq(model="qwen/qwen3-32b")
# async def scrape_reddit_topics(topics:List[str]) -> dict[str,str]:
#     """ Process list of topics and return analysis results """
#     # read write sessions with uper walay server params
#     # mcp setup
#     async with stdio_client(server_params) as (read,write):
#         async with ClientSession(read,write) as session:
#             await session.initialize()
#             tools = await load_mcp_tools(session)
#             # yeh bana h agent langraph sy
#             agent = create_react_agent(model,tools)

#             reddit_results = {}
#             for topic in topics:
#                 # realtime summary agent jo dey rha h wo yahan save hori h
#                 summary=await process_topic(agent,model)
#                 reddit_results[topic] = summary
#                 await asyncio.sleep(5) # rate limiting

#             return{"reddit analysis":reddit_results}
        
#         #Salesforce, Mulesoft, Service Now, Informatica which are all extremely popular right now
# # MIS. Learn how to do system integration planning and business process analysis without really having any real technical expertise. It pays well also.



# mcp_limiter = AsyncLimiter(1,15)
# #max rate,time

# two_weeks_ago = datetime.today() - timedelta(days=14)
# two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')

# async def process_topic(agent,topic:str):
#     async with mcp_limiter:
#         messages= [
#             {
#                 "role":"system",
#                 "content":f''' You are a reddit analysis expert.Use available tools to:
#                 1.Find top 2 posts about the given topic BUT only after {two_weeks_ago_str}, NOTHING before this date strictly!
#                 2. Analyze their content and sentiment
#                 3. Create a summary of discussion and overall sentiment
#                 '''    
#             },
#             {
#                 "role":"user",
#                 "content":f"""Analyze Reddit posts about '{topic}'.Provide a comprehensive summary including:
#                 - Main description points.
#                 - Key opinions expressed
#                 - Any notable trends or patterns
#                 - Summarize the overall narrative, discussion points and also quote interesting comments without mentioning names
#                 - Overall sentiment (positive/neutral/negative)"""
#             }
#         ]

#         try:
#             response = await agent.ainvoke({"messages":messages})
#             return response["messages"][-1].content
#         except Exception as e:
#             raise e


from typing import List
import os
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Setting up server parameters of mcp
server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("BRIGHTDATA_API_TOKEN"),
        "RATE_LIMIT": "100/1h",
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"]
)

# Initialize HuggingFace model for LangChain
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_API_KEY"),
    temperature=0.3,
    max_new_tokens=2000
)
model = ChatHuggingFace(llm=llm)

mcp_limiter = AsyncLimiter(1, 15)
two_weeks_ago = datetime.today() - timedelta(days=14)
two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')


async def scrape_reddit_topics(topics: List[str]) -> dict:
    """Process list of topics and return analysis results"""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools)

                reddit_results = {}
                for topic in topics:
                    try:
                        summary = await process_topic(agent, topic)
                        reddit_results[topic] = summary
                    except Exception as e:
                        print(f"Error processing topic {topic}: {str(e)}")
                        reddit_results[topic] = f"Error analyzing Reddit for {topic}"
                    
                    await asyncio.sleep(5)  # rate limiting

                return {"reddit_analysis": reddit_results}
    except Exception as e:
        print(f"Reddit scraping failed: {str(e)}")
        # Return empty results instead of crashing
        return {"reddit_analysis": {topic: f"Reddit analysis unavailable" for topic in topics}}


async def process_topic(agent, topic: str):
    """Process a single topic using the agent"""
    async with mcp_limiter:
        messages = [
            {
                "role": "system",
                "content": f'''You are a reddit analysis expert. Use available tools to:
                1. Find top 2 posts about the given topic BUT only after {two_weeks_ago_str}, NOTHING before this date strictly!
                2. Analyze their content and sentiment
                3. Create a summary of discussion and overall sentiment
                '''
            },
            {
                "role": "user",
                "content": f"""Analyze Reddit posts about '{topic}'. Provide a comprehensive summary including:
                - Main description points.
                - Key opinions expressed
                - Any notable trends or patterns
                - Summarize the overall narrative, discussion points and also quote interesting comments without mentioning names
                - Overall sentiment (positive/neutral/negative)"""
            }
        ]

        try:
            response = await agent.ainvoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            print(f"Error in process_topic: {str(e)}")
            return f"Error analyzing this topic: {str(e)}"