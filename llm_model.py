import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

def initialize_llm():
    load_dotenv()

    api_key = st.secrets["openrouter"]["api_key"]
    if not api_key:
        raise ValueError("OPENROUTER_KEY not found in environment variables!")
    
    return ChatOpenAI(
        model="tngtech/deepseek-r1t2-chimera:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_retries=2,
    )
