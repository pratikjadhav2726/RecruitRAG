import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import ColdEmailGraph
from portfolio import Portfolio
from utils import clean_text

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
                # Load and clean website data
                loader = WebBaseLoader([url_input])
                scraped_text = clean_text(loader.load().pop().page_content)
                
                # Initialize graph
                cold_mail_graph = ColdEmailGraph()
                compiled_graph = cold_mail_graph.build_graph()
                
                # Run through LangGraph
                result = compiled_graph.invoke({"scraped_text": scraped_text})
                
                # Display results
                emails = result.get("emails", [])
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

            except Exception as e:
                st.error(f"🚨 An error occurred: `{e}`")