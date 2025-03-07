import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import ColdEmailGraph
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app():
    st.title("📧 Cold Mail Generator with LangGraph")
    url_input = st.text_input("Enter a URL:", value="https://jobs.nike.com/job/R-33460")
    submit_button = st.button("Submit")

    if submit_button:
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
            for email in result.get("emails", []):
                st.code(email, language="markdown")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")

if __name__ == "__main__":
    create_streamlit_app()


