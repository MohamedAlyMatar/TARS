import requests
import json
import streamlit as st
import pdfplumber
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

# for local use uncomment this to Load environment variables from .env file
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please check your '.env' file.")
client = OpenAI(api_key=api_key)

# Google CSE API key and Search Engine ID
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

def parse_pdf(file):
    """
    Extract text content from a PDF file.
    """
    parsed_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            parsed_text.append(page.extract_text())
    return "\n".join(parsed_text)


def google_cse_search(query, num_results=10):
    """
    Search LinkedIn profiles using Google Custom Search Engine.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": min(num_results, 10),  # Max 10 results per request
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        print(response.json().get("items", []))
        return response.json().get("items", [])
    else:
        st.error(f"Error with Google CSE API: {response.status_code} - {response.text}")
        return []

def validate_profile_with_llm(profile_description, custom_criteria):
    """
    Use the LLM to validate if a profile's experience aligns with the desired expertise level.
    """
    messages = [
        {"role": "system", "content": """
            You are a professional Talent Acquisition Specialist. 
            Your task is to evaluate a LinkedIn profile description based on the provided criteria.
            If the profile is an excellent match, respond with 'Valid profile'.
            Otherwise, respond with 'Invalid profile'.
        """},
        {"role": "user", "content": f"Criteria: {custom_criteria}"},
        {"role": "user", "content": f"Profile Description:\n{profile_description}"}
    ]

    try:
        # Query the LLM for validation
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=50
        )
        response = chat_completion.choices[0].message.content.strip()
        return response == "Valid profile"
    except Exception as e:
        st.error(f"Error validating profile with LLM: {e}")
        return False

# Integrate LLM validation into the profile filtering workflow
def get_top_profiles_with_validation(profiles, custom_criteria, top_n=10):
    """
    Validate and filter profiles using LLM before selecting the top N matches.
    """
    validated_profiles = []

    for profile in profiles:
        profile_description = profile.get("description", "")
        if validate_profile_with_llm(profile_description, custom_criteria):
            validated_profiles.append(profile)

    # Sort and select the top N profiles (sorting logic can be adjusted based on profile relevance)
    validated_profiles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return validated_profiles[:top_n]


# Streamlit UI
st.title("TARS")
st.subheader("Resume Matching and LinkedIn Search")

# Input prompt for the LLM
# custom_prompt = st.text_input("Enter your matching criteria prompt:", placeholder="Describe the skills or criteria...")

# Input for LinkedIn profile search
skills = st.text_input("Enter specific skills to search LinkedIn profiles:", placeholder="e.g., Python, Data Science")
country = st.selectbox("Select the country for LinkedIn profile search:", ["", "Cairo","United States", "India", "Canada", "Other"])

# File uploader for resumes
# uploaded_files = st.file_uploader("Upload Resumes (PDFs only)", type=["pdf"], accept_multiple_files=True)

# Initialize session state for extracted data
if "resume_data" not in st.session_state:
    st.session_state["resume_data"] = pd.DataFrame(columns=["Name", "Title", "Phone number", "LinkedIn link"])

if "linkedin_results" not in st.session_state:
    st.session_state["linkedin_results"] = pd.DataFrame(columns=["Title", "Link"])

# Button to search LinkedIn profiles
if st.button("Search LinkedIn Profiles"):
    if skills:
        query = f"site:linkedin.com/in/ {skills}"
        if country:
            query += f" {country}"
        results = google_cse_search(query, num_results=5)
        if results:
            linkedin_data = [{"Title": item["title"], "Link": item["link"]} for item in results]
            st.session_state["linkedin_results"] = pd.DataFrame(linkedin_data)
        else:
            st.warning("No LinkedIn profiles found.")
    else:
        st.warning("Please enter skills to search.")

# Display LinkedIn search results
if not st.session_state["linkedin_results"].empty:
    st.subheader("LinkedIn Profiles")
    st.table(st.session_state["linkedin_results"])
