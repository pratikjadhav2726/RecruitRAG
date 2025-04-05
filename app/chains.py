import os
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from portfolio import Portfolio
from utils import clean_text
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st
from typing import Dict, List, Union,Any
from pydantic import BaseModel, ValidationError

# Load environment variables
load_dotenv()


class JobPosting(BaseModel):
    role: str
    experience: str
    skills: List[str] = []
    description: str = None

class ColdEmailGraph:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.3-70b-versatile"
        )
        self.portfolio = Portfolio()

    def extract_jobs(self, state):
        """Extract job postings from the scraped text."""
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": state['scraped_text']})
        try:
            json_parser = JsonOutputParser()
            jobs = json_parser.parse(res.content)
            # print(jobs)
            validated_jobs = [JobPosting(**job).dict() for job in (jobs if isinstance(jobs, list) else [jobs])]
            # print("v",validated_jobs)
        except (OutputParserException, ValidationError) as e:
            print(f"Parsing/Validation error: {e}")
            return {**state, "jobs": []}
        return {**state,"jobs": validated_jobs}

    def check_coherence(self, state):
        """Check coherence of the extracted job descriptions."""
        jobs = state["jobs"]
        coherent_jobs = [job for job in jobs if "role" in job and "description" in job]
        coherence_score = len(coherent_jobs) / max(len(jobs), 1)  # Simple ratio-based score
        # print(coherence_score)

        return {**state, "jobs": coherent_jobs, "coherence_score": coherence_score}

    def retrieve_links(self, state):
        """Retrieve relevant portfolio links based on extracted skills."""
        jobs = state["jobs"]
        for job in jobs:
            job["links"] = self.portfolio.query_links(job.get("skills", []))
        return {**state,"jobs": jobs}
    
    def check_rag_score(self, state):
        """Filter jobs based on RAG score (relevance of retrieved links)."""
        jobs = state["jobs"]
        filtered_jobs = [job for job in jobs if len(job.get("links", [])) > 0]
        rag_score = len(filtered_jobs) / max(len(jobs), 1)  # Simple ratio-based score

        return {**state, "jobs": filtered_jobs, "rag_score": rag_score}

    def write_mail(self, state):
        """Generate a cold email for the extracted job and relevant portfolio links."""
        jobs = state["jobs"]
        emails = []
        for job in jobs:
            prompt_email = PromptTemplate.from_template(
                """
                ### JOB DESCRIPTION:
                {job_description}
                ### INSTRUCTION:
                You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
                the seamless integration of business processes through automated tools.
                Your job is to write a cold email to the client regarding the job mentioned above, describing AtliQ’s capability in fulfilling their needs.
                Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
                Remember you are Mohan, BDE at AtliQ.
                Do not provide a preamble.
                ### EMAIL (NO PREAMBLE):
                """
            )
            chain_email = prompt_email | self.llm
            res = chain_email.invoke({"job_description": str(job), "link_list": job["links"]})
            emails.append(res.content)
        return {"emails": emails}
    def log_wrapper(self,node_name, node_func):
        def wrapped(state: Dict[str, Any]):
            print(f"Executing Node: {node_name}")
            new_state = node_func(state)
            # print(f"Node '{node_name}' Output State:", new_state)
            return new_state
        return wrapped
    def build_graph(self):
        """Construct the LangGraph pipeline with conditional retries."""
        graph = StateGraph(Dict[str, Any])  # Define state type
    
        # Add nodes
        graph.add_node("extract_jobs", self.log_wrapper("extract_jobs", self.extract_jobs))
        graph.add_node("check_coherence", self.log_wrapper("check_coherence", self.check_coherence))
        graph.add_node("retrieve_links", self.log_wrapper("retrieve_links", self.retrieve_links))
        graph.add_node("check_rag_score", self.log_wrapper("check_rag_score", self.check_rag_score))
        graph.add_node("write_mail", self.log_wrapper("write_mail", self.write_mail))

        # Define main flow
        graph.set_entry_point("extract_jobs")
        graph.add_edge("extract_jobs", "check_coherence")

        # Conditional Edge: Retry if coherence_score < 0.8
        graph.add_conditional_edges(
            "check_coherence",
            lambda state: "extract_jobs" if state["coherence_score"] < 0.8 else "retrieve_links"
        )

        graph.add_edge("retrieve_links", "check_rag_score")

        # Conditional Edge: Retry if rag_score < 0.8
        graph.add_conditional_edges(
            "check_rag_score",
            lambda state: "retrieve_links" if state["rag_score"] < 0.8 else "write_mail"
        )

        return graph.compile()