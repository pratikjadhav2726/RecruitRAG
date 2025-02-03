import streamlit as st
from agent import AgenticChain
from portfolio import Portfolio
from utils import clean_text

st.title("📧 Agentic RAG Cold Email Generator")

agent_chain = AgenticChain()

url_input = st.text_input("Enter Job Posting URL")
submit_button = st.button("Submit")

if submit_button:
    try:
        job_text = agent_chain.run_agent(f"Scrape job details from this URL: {url_input} and with that all scraped data extract job details")
        # st.write("**Extracted Job Description:**", job_text)
        # print("hi1",type(job_text),job_text)
        # portfolio= Portfolio()
        # st.write("hi1",job_text)
        # job_details = agent_chain.run_agent(f"{job_text}")
        # st.write("**Parsed Job Details:**", job_details)
        # print("Here are the job details")
        # skills = job_details.get('skills', [])
        # links = portfolio.query_links(skills)
        # st.write("Here are the links for portfolio",links)

        email = agent_chain.run_agent(f"Generate a cold email for this job: {job_text}")
        st.write("**Generated Cold Email:**", job_text)

        # # feedback = st.text_area("Enter Recruiter Feedback")
        # if st.button("Refine Email"):
        #     refined_email = agent_chain.run_agent(f"Refine this email: {email}\nFeedback: {feedback}")
        #     st.write("**Refined Cold Email:**", refined_email)

    except Exception as e:
        st.error(f"An Error Occurred: {e}")
