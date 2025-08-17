import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def initialize_llm():
    # Load .env first
    load_dotenv()
    
    # Use the OpenRouter key
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_KEY not found in environment variables!")
    
    return ChatOpenAI(
        model="tngtech/deepseek-r1t2-chimera:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_retries=2,
    )
