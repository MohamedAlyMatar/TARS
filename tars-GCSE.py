import requests
import json
import streamlit as st
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

# ----------------------------------------------------------------------------------- KEYS
## for local use uncomment this to Load environment variables from .env file
# load_dotenv(override=True)
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("API key not found. Please check your '.env' file.")
# client = OpenAI(api_key=api_key)

# Google CSE API key and Search Engine ID
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = OpenAI(api_key = st.secrets["openai"]["api_key"])
GOOGLE_API_KEY   = st.secrets["GOOGLE_API_KEY"]
SEARCH_ENGINE_ID = st.secrets["SEARCH_ENGINE_ID"]

# ----------------------------------------------------------------------------------- LLM
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

def rank_profiles_with_llm(profiles, prompt):
    """
    Rank LinkedIn profiles using GPT-4o based on a custom prompt.
    """
    messages = [
        {"role": "system", "content": """
            You are a professional Talent Acquisition Specialist.
            Your task is to rank LinkedIn profiles based on their relevance to a given prompt.
            Provide a numerical score (1-100) for each profile.
        """},
        {"role": "user", "content": f"Criteria: {prompt}"}
    ]

    ranked_profiles = []
    for profile in profiles:
        profile_description = (
            profile.get("pagemap", {})  # Get 'pagemap' dictionary or return an empty dictionary
            .get("metatags", [{}])[0]  # Get the first element of 'metatags' or a default empty dict
            .get("og:description", "")  # Get 'og:description' or return an empty string
)        # profile_description = profile.get("og:description", "")
        print("\nProfile Description:", profile_description)
        if not profile_description:
            continue

        messages.append({"role": "user", "content": f"Profile Description:\n{profile_description}"})
        try:
            # Query GPT-4o for scoring
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=50
            )
            response = chat_completion.choices[0].message.content.strip()

            # Extract the numerical score from GPT response
            match = re.search(r"\b\d+\b", response)  # Match any standalone number
            if match:
                score = int(match.group())  # Convert matched number to integer
            else:
                score = 0  # Default score if no number is found in response

            ranked_profiles.append({"profile": profile, "score": score})

        except Exception as e:
            st.error(f"Error ranking profile with LLM: {e}")
        
        # Remove the added user message to reset for the next profile
        messages.pop()

    return sorted(ranked_profiles, key=lambda x: x["score"], reverse=True)

# ----------------------------------------------------------------------------------- Google CSE
def google_cse_search(query, num_results=100):
    """
    Search LinkedIn profiles using Google Custom Search Engine.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": min(num_results, 10),  # Fetch 10 results per request
    }
    all_results = []
    for start in range(1, num_results + 1, 10):
        params["start"] = start
        response = requests.get(url, params=params)
        if response.status_code == 200:
            items = response.json().get("items", [])
            all_results.extend(items)
            if len(items) < 10:  # If fewer results are returned, stop early
                break
        else:
            st.error(f"Error with Google CSE API: {response.status_code} - {response.text}")
            break
    # print(all_results)
    return all_results[:num_results]  # Ensure we cap results at `num_results`

# Combine search, ranking, and filtering
def get_top_profiles(query, prompt, num_results=100, top_n=10):
    """
    Search for LinkedIn profiles, rank them using GPT-4o, and return the top N profiles.
    """
    st.info("Searching for profiles...")
    profiles = google_cse_search(query, num_results=num_results)
    if not profiles:
        st.warning("No profiles found.")
        return []

    st.info("Ranking profiles...")
    ranked_profiles = rank_profiles_with_llm(profiles, prompt)

    # Filter top N profiles
    top_profiles = ranked_profiles[:top_n]
    return top_profiles


# ----------------------------------------------------------------------------------- Streamlit UI
st.title("TARS")
st.subheader("Resume Matching and LinkedIn Search")

# Input fields
skills = st.text_input("Enter specific skills to search LinkedIn profiles:", placeholder="e.g., Python, Data Science")
country = st.selectbox("Select the country for LinkedIn profile search:", ["", "Cairo","United States", "India", "Canada", "Other"])
# prompt = st.text_input("Enter criteria for ranking profiles:", placeholder="e.g., Data Scientist with Python and ML experience")
custom_prompt = st.text_input("Enter your matching criteria prompt:", placeholder="Describe the skills or criteria...")

# Initialize session state for extracted data
if "resume_data" not in st.session_state:
    st.session_state["resume_data"] = pd.DataFrame(columns=["Name", "Title", "Phone number", "LinkedIn link"])

if "linkedin_results" not in st.session_state:
    st.session_state["linkedin_results"] = pd.DataFrame(columns=["Title", "Link"])


# # Streamlit UI Integration
if st.button("Search LinkedIn Profiles"):
    if skills:
        query = f"site:linkedin.com/in/ {skills}"
        if country:
            query += f" {country}"
        if custom_prompt:
            top_profiles = get_top_profiles(query, custom_prompt, num_results=100, top_n=10)
            if top_profiles:
                profile_table = []
                for i, profile in enumerate(top_profiles, start=1):
                    profile_data = profile["profile"]
                    profile_table.append({
                        "Rank": i,
                        "Title": profile_data.get("title", "No Title"),
                        "Link": f"[Profile]({profile_data.get('link', '#')})",
                        # "Description": profile_data.get("snippet", "No Description"),
                        "Relevance Score": profile.get("score", 0)
                    })

                st.session_state["linkedin_results"] = pd.DataFrame(profile_table)
                st.write("### Top LinkedIn Profiles")
                st.table(st.session_state["linkedin_results"])

if st.button("Export Results to CSV"):
    if not st.session_state["linkedin_results"].empty:
        csv_data = st.session_state["linkedin_results"].to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="linkedin_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("No results available to export.")
