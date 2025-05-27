"""
Streamlit web application for the Agentic RAG Cold Email Generator.

This application provides a user interface for inputting a job URL,
processing it through a LangGraph-based chain to extract job details,
retrieve relevant portfolio links, and generate personalized cold emails.

The main components are:
- Streamlit UI elements for input and display.
- WebBaseLoader for fetching content from the provided URL.
- `clean_text` utility for preprocessing the scraped HTML.
- `ColdEmailGraph` from `chains.py` for the core logic.
- Error handling for network requests and content processing.
"""
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import ColdEmailGraph
from portfolio import Portfolio
from utils import clean_text
import logging
from requests.exceptions import RequestException

# Basic Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set Streamlit page config
st.set_page_config(page_title="Cold Email Generator", page_icon="📧", layout="centered")

st.title("📧 Agentic RAG Cold Email Generator")
st.markdown("Generate personalized cold emails from career pages using AI, RAG, and LangGraph 🔗")

# Sidebar with instructions
with st.sidebar:
    st.header("📝 Instructions")
    st.write("""
    1. Paste a job URL from a company careers page.
    2. Click **Submit** to generate a cold email.
    3. Emails will include portfolio links relevant to the job.
    """)
    st.markdown("---")
    st.caption("Developed by [Pratik Jadhav](https://www.linkedin.com/in/pratikjadhav2726)")

# Input form
st.markdown("### 🔗 Paste a careers/job URL:")
url_input = st.text_input("Job URL", value="https://jobs.nike.com/job/R-33460")
submit_button = st.button("🚀 Submit")

if submit_button:
    if url_input.strip() == "":
        st.warning("⚠️ Please enter a valid URL.")
    else:
        with st.spinner("🔍 Processing job data and generating cold email..."):
            try:
                # Step 1: Load and clean website data from the provided URL
                loader = WebBaseLoader([url_input]) # Initialize WebBaseLoader with the URL
                # Attempt to load content; WebBaseLoader returns a list of Documents
                # We expect one document for one URL.
                document = loader.load()
                if not document:
                    st.error("🚨 Failed to load content from the URL. The page might be inaccessible or empty.")
                    return
                
                scraped_text = clean_text(document[0].page_content) # Clean the HTML content of the first document

                # Step 2: Check if scraped text is empty after cleaning
                if not scraped_text:
                    st.warning("⚠️ Scraped content is empty after cleaning. Cannot proceed with email generation.")
                    return # Stop execution if no content to process
                
                # Step 3: Initialize and run the ColdEmailGraph
                cold_mail_graph = ColdEmailGraph() # Instantiate the graph
                compiled_graph = cold_mail_graph.build_graph() # Compile the graph structure
                
                # Invoke the graph with the scraped text. The state starts with 'scraped_text'.
                result = compiled_graph.invoke({"scraped_text": scraped_text})
                
                # Step 4: Process and display results
                emails = result.get("emails", []) # Get generated emails from the final state
                jobs = result.get("jobs", [])

                if not emails:
                    st.warning("❌ No emails generated. The page may not contain valid job info.")
                else:
                    for i, email in enumerate(emails):
                        st.markdown(f"---\n### 📬 Cold Email {i+1}")
                        st.code(email, language="markdown")

                        if i < len(jobs):
                            job = jobs[i]
                            with st.expander("📋 View Extracted Job Details"):
                                st.markdown(f"**Role:** {job.get('role', 'N/A')}")
                                st.markdown(f"**Experience:** {job.get('experience', 'N/A')}")
                                skills = job.get("skills", [])
                                if skills:
                                    st.markdown("**Skills:**")
                                    st.markdown(", ".join(skills))
                                links = job.get("links", [])
                                if links:
                                    st.markdown("**Portfolio Links:**")
                                    for link in links:
                                        st.markdown(f"- [{link}]({link})")

            except RequestException as e:
                # Specific handling for network-related errors during URL loading (e.g., DNS failure, connection timeout)
                st.error(f"🚨 Network error loading URL: {e}")
                logging.error(f"RequestException: {e} for URL: {url_input}")
            except IndexError as e:
                # Handles errors if `loader.load()` returns an empty list or accessing `document[0]` fails.
                # This might also catch issues if `clean_text` fails unexpectedly on certain inputs,
                # though it's less likely to raise IndexError itself.
                st.error(f"🚨 Error processing scraped content (e.g., unexpected document structure or empty content): {e}")
                logging.error(f"IndexError: {e} processing URL: {url_input}")
            except Exception as e: # General fallback for any other unexpected errors
                st.error(f"🚨 An unexpected error occurred: {e}")
                logging.exception(f"Unexpected error processing URL {url_input}: {e}") # Log full traceback