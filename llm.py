# from .config import API_KEY
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os

# for local use uncomment this to Load environment variables from .env file
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please check your '.env' file.")
# client = OpenAI(api_key=api_key)

# client = OpenAI(api_key = st.secrets["openai"]["api_key"])

class LLM:
    def __init__(self)  -> None:
        self.client = OpenAI(api_key=api_key)
        # self.client = OpenAI(api_key = st.secrets["openai"]["api_key"])

    def call(self, list_of_messages : list) -> str:
        '''
        list of system and user prompts

        Call the LLM, do what you want, only return text

        return str

        '''
        
        chat_completion = self.client.chat.completions.create(messages=list_of_messages, model="gpt-4o-mini")
        
        return chat_completion.choices[0].message.content

llm = LLM()