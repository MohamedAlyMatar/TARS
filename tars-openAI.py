import json
import streamlit as st
import pdfplumber
import re
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

def parse_pdf(file):
    """
    Extract text content from a PDF file.
    """
    parsed_text = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            parsed_text.append(page.extract_text())
    return "\n".join(parsed_text)

# Load environment variables from .env file
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please check your '.env' file.")
client = OpenAI(api_key=api_key)

def process_with_llm(resume_text, custom_prompt):
    """
    Process the resume text with the LLM to identify relevant information.
    """
    messages = [
        {"role": "system", "content": """
            You are a professional Talent Acquisition and Recruitment Specialist. 
            Your task is to review the provided resume data and match it with the user's criteria.
            If the resume is a good candidate based on the criteria, WITHOUT ANY EXPLANATION, ONLY return a csv LINE containing only:
            Name (e.g., John Doe),Title (e.g., Software Engineer),Phone number (e.g., +123456789),LinkedIn link (e.g., https://www.linkedin.com/in/johndoe)
            If the resume does not match, respond with 'No match found.'"""},
        {"role": "user", "content": f"Criteria: {custom_prompt}"},
        {"role": "user", "content": f"Resume:\n{resume_text}"}
    ]
    
    try:
        # Query the LLM for processing
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        st.error(f"Error processing resume with LLM: {e}")
        return None


def extract_information_from_response(llm_response):
    """
    Parse CSV from the LLM response and extract relevant fields:
    Name, Title, Phone number, and LinkedIn link.
    """
    if "No match found" in llm_response.strip():
        st.warning("No match found for the provided criteria.")
        return None

    try:
        # Split the response into lines and remove extra spaces
        lines = [line.strip() for line in llm_response.strip().split("\n") if line.strip()]
        
        if len(lines) < 1:
            st.warning("No valid data found.")
            return None

        # Extract header and data if there's more than one line
        headers = ["Name", "Title", "Phone number", "LinkedIn link"]

        extracted_data_list = []
        for line in lines:
            # Split by commas but handle cases with quotes around text fields
            data = [field.strip().replace('"', '') for field in line.split(",")]

            # Handle cases where the data might not have all fields, set missing fields to None
            if len(data) == len(headers):
                extracted_data = {headers[i]: data[i] for i in range(len(headers))}
                extracted_data_list.append(extracted_data)
            else:
                st.warning(f"Data mismatch found in line: {line}")
        
        return extracted_data_list  # Return a list of extracted data dictionaries
    except Exception as e:
        st.error(f"Error processing CSV from response: {e}")
        return None


# Streamlit UI
st.title("TARS")
st.subheader("Resume Matching and Data Extraction with LLM")

# Input prompt for the LLM
custom_prompt = st.text_input("Enter your matching criteria prompt:", placeholder="Describe the skills or criteria...")

# File uploader for resumes
uploaded_files = st.file_uploader("Upload Resumes (PDFs only)", type=["pdf"], accept_multiple_files=True)

# Initialize session state for extracted data
if "resume_data" not in st.session_state:
    st.session_state["resume_data"] = pd.DataFrame(columns=["Name", "Title", "Phone number", "LinkedIn link"])

# Button to trigger resume processing
if st.button("Process Resumes"):
    # Clear existing data before processing
    st.session_state["resume_data"] = pd.DataFrame(columns=["Name", "Title", "Phone number", "LinkedIn link"])

    if uploaded_files and custom_prompt:
        for file in uploaded_files:
            # Parse PDF content
            resume_text = parse_pdf(file)

            # Process with LLM
            llm_response = process_with_llm(resume_text, custom_prompt)

            if llm_response:
                # Extract structured data from LLM response
                resume_info = extract_information_from_response(llm_response)

                if resume_info:
                    # Add extracted data to the DataFrame
                    new_data = pd.DataFrame(resume_info)
                    st.session_state["resume_data"] = pd.concat([st.session_state["resume_data"], new_data], ignore_index=True)

        # Display table of extracted information
        if not st.session_state["resume_data"].empty:
            st.subheader("Matching Resumes")
            st.table(st.session_state["resume_data"])
        else:
            st.warning("No matching candidates found.")
    else:
        if not custom_prompt:
            st.warning("Please enter a matching criteria prompt.")
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
