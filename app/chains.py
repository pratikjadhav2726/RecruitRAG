import os
import logging
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

# Configuration Constants for ColdEmailGraph
DEFAULT_COHERENCE_THRESHOLD = 0.8
DEFAULT_RAG_SCORE_THRESHOLD = 0.8
SENDER_PERSONA_NAME = "Mohan"
SENDER_COMPANY_NAME = "AtliQ"
SENDER_COMPANY_DESCRIPTION = "an AI & Software Consulting company dedicated to facilitating the seamless integration of business processes through automated tools."

class JobPosting(BaseModel):
    """
    Pydantic model representing a single job posting extracted from a website.

    Attributes:
        role (str): The title or role of the job.
        experience (str): Required or preferred experience level for the job.
        skills (List[str], optional): A list of skills relevant to the job. Defaults to an empty list.
        description (str, optional): A summary or description of the job. Defaults to None.
    """
    role: str
    experience: str
    skills: List[str] = []
    description: str = None

class ColdEmailGraph:
    """
    Implements a LangGraph-based workflow for generating cold emails.

    The graph processes scraped job text through several stages:
    1.  Job Extraction: Extracts structured job postings from text.
    2.  Coherence Check: Validates the quality of extracted jobs.
    3.  Link Retrieval: Fetches relevant portfolio links based on job skills.
    4.  RAG Score Check: Filters jobs based on the relevance of retrieved links.
    5.  Email Generation: Writes personalized cold emails for valid jobs.

    The graph includes conditional edges to retry extraction or link retrieval
    if quality scores (coherence, RAG score) are below defined thresholds.

    Attributes:
        llm (ChatGroq): Instance of the Groq LLM used for text generation tasks.
        portfolio (Portfolio): Instance of the Portfolio class for querying project links.
        state (Dict[str, Any]): The shared state dictionary passed between graph nodes.
            Expected keys at various stages:
            - 'scraped_text': Initial input text from job page.
            - 'jobs': List of extracted JobPosting objects.
            - 'coherence_score': Score indicating quality of job extraction.
            - 'rag_score': Score indicating relevance of retrieved portfolio links.
            - 'emails': List of generated cold email strings.
    """
    def __init__(self):
        """
        Initializes the ColdEmailGraph, setting up the LLM and Portfolio instances.
        The LLM model name is configurable via the `GROQ_LLM_MODEL_NAME` environment
        variable, defaulting to "llama-3.3-70b-versatile".
        """
        default_model_name = "llama-3.3-70b-versatile"
        llm_model_name = os.getenv("GROQ_LLM_MODEL_NAME", default_model_name)
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=llm_model_name
        )
        logging.info(f"Using Groq LLM model: {llm_model_name}") # Added logging
        self.portfolio = Portfolio()
        # The 'state' is implicitly managed by LangGraph but conceptualized here for clarity.

    def extract_jobs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts job postings from the scraped text using an LLM chain.

        Args:
            state (Dict[str, Any]): The current graph state.
                Expects 'scraped_text' key with the raw text from a job page.

        Returns:
            Dict[str, Any]: The updated state dictionary.
                Adds 'jobs' key with a list of validated JobPosting dictionaries.
                If parsing or validation fails, 'jobs' will be an empty list.
        """
        logging.info("Node: extract_jobs - Extracting job postings from scraped text.")
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
            logging.warning(f"Parsing/Validation error in extract_jobs: {e}. Returning empty job list.")
            return {**state, "jobs": []}
        return {**state, "jobs": validated_jobs}

    def check_coherence(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks the coherence of extracted job descriptions.

        A job is considered coherent if it has both 'role' and 'description' fields.
        The coherence score is the ratio of coherent jobs to the total number of jobs.

        Args:
            state (Dict[str, Any]): The current graph state.
                Expects 'jobs' key with a list of JobPosting dictionaries.

        Returns:
            Dict[str, Any]: The updated state dictionary.
                Updates 'jobs' to contain only coherent jobs.
                Adds 'coherence_score' key with the calculated score.
        """
        logging.info("Node: check_coherence - Checking coherence of extracted jobs.")
        jobs = state.get("jobs", [])
        if not jobs: # If no jobs were extracted, coherence is effectively 0
            return {**state, "jobs": [], "coherence_score": 0.0}

        coherent_jobs = [job for job in jobs if job.get("role") and job.get("description")]
        coherence_score = len(coherent_jobs) / len(jobs) if jobs else 0.0
        
        logging.info(f"Coherence check: {len(coherent_jobs)} coherent jobs out of {len(jobs)}. Score: {coherence_score}")
        return {**state, "jobs": coherent_jobs, "coherence_score": coherence_score}

    def retrieve_links(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves relevant portfolio links for each coherent job based on its skills.

        Args:
            state (Dict[str, Any]): The current graph state.
                Expects 'jobs' key with a list of coherent JobPosting dictionaries.

        Returns:
            Dict[str, Any]: The updated state dictionary.
                Each job dictionary in the 'jobs' list is updated with a 'links' key,
                containing a list of relevant portfolio link metadata.
        """
        logging.info("Node: retrieve_links - Retrieving portfolio links for jobs.")
        jobs = state.get("jobs", [])
        for job in jobs:
            # Query for relevant links using skills from the job posting.
            # The number of results can be configured in portfolio.query_links
            job["links"] = self.portfolio.query_links(skills=job.get("skills", []))
        return {**state, "jobs": jobs}
    
    def check_rag_score(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters jobs based on a RAG (Retrieval-Augmented Generation) score.

        The RAG score is determined by the presence of retrieved portfolio links.
        Jobs without any relevant links are filtered out. The score is the ratio
        of jobs with links to the total number of jobs processed in this step.

        Args:
            state (Dict[str, Any]): The current graph state.
                Expects 'jobs' key with a list of JobPosting dictionaries,
                each potentially updated with 'links'.

        Returns:
            Dict[str, Any]: The updated state dictionary.
                Updates 'jobs' to contain only jobs that have associated links.
                Adds 'rag_score' key with the calculated score.
        """
        logging.info("Node: check_rag_score - Checking RAG score based on retrieved links.")
        jobs = state.get("jobs", [])
        if not jobs: # If no jobs to check, RAG score is effectively 0
            return {**state, "jobs": [], "rag_score": 0.0}
            
        filtered_jobs = [job for job in jobs if job.get("links")] # A job passes if it has any links
        rag_score = len(filtered_jobs) / len(jobs) if jobs else 0.0
        
        logging.info(f"RAG score check: {len(filtered_jobs)} jobs with links out of {len(jobs)}. Score: {rag_score}")
        return {**state, "jobs": filtered_jobs, "rag_score": rag_score}

    def write_mail(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a personalized cold email for each job that has passed all checks.

        Args:
            state (Dict[str, Any]): The current graph state.
                Expects 'jobs' key with a list of filtered JobPosting dictionaries,
                each containing 'links' for portfolio items.

        Returns:
            Dict[str, Any]: The final state dictionary.
                Adds 'emails' key with a list of generated email content strings.
        """
        logging.info("Node: write_mail - Generating cold emails.")
        jobs = state.get("jobs", [])
        emails = []
        for job in jobs:
            # The prompt uses sender persona constants defined at the module level.
            email_prompt_text = f"""
                ### JOB DESCRIPTION:
                {{job_description}}
                ### INSTRUCTION:
                You are {SENDER_PERSONA_NAME}, a business development executive at {SENDER_COMPANY_NAME}. {SENDER_COMPANY_NAME} is {SENDER_COMPANY_DESCRIPTION}.
                Your job is to write a cold email to the client regarding the job mentioned above, describing {SENDER_COMPANY_NAME}’s capability in fulfilling their needs.
                Also add the most relevant ones from the following links to showcase {SENDER_COMPANY_NAME}'s portfolio: {{link_list}}
                Remember you are {SENDER_PERSONA_NAME}, BDE at {SENDER_COMPANY_NAME}.
                Do not provide a preamble.
                ### EMAIL (NO PREAMBLE):
                """
            prompt_email = PromptTemplate.from_template(email_prompt_text)
            chain_email = prompt_email | self.llm
            # Invoke LLM to generate email based on job details and curated links
            res = chain_email.invoke({
                "job_description": str(job), # Pass the full job details as a string
                "link_list": job.get("links", []) # Pass the list of portfolio links
            })
            emails.append(res.content)
        # This node outputs only the emails, effectively ending this branch of the graph's state progression for 'jobs'.
        return {"emails": emails} 

    def log_wrapper(self, node_name: str, node_func: callable) -> callable:
        """
        A decorator that wraps graph nodes to log their execution and state.

        Args:
            node_name (str): The name of the node being wrapped.
            node_func (callable): The actual function (node) to be executed.

        Returns:
            callable: The wrapped function which includes logging.
        """
        def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            # This log message is now handled by the specific node functions for better context.
            # logging.info(f"Executing Node: {node_name}") 
            new_state = node_func(state)
            logging.debug(f"Node '{node_name}' output state: {new_state}")
            return new_state
        return wrapped

    def build_graph(self) -> StateGraph:
        """
        Constructs and compiles the LangGraph pipeline with nodes and conditional edges.

        The graph defines the flow of operations:
        extract_jobs -> check_coherence
                       (if coherent) -> retrieve_links -> check_rag_score
                                                          (if good RAG score) -> write_mail
                                                          (else, retry retrieve_links)
                       (else, retry extract_jobs)

        Returns:
            StateGraph: The compiled LangGraph object ready for execution.
        """
        graph = StateGraph(Dict[str, Any])  # Define the type of the shared state
    
        # Add nodes to the graph, wrapping each with the logging decorator
        graph.add_node("extract_jobs", self.log_wrapper("extract_jobs", self.extract_jobs))
        graph.add_node("check_coherence", self.log_wrapper("check_coherence", self.check_coherence))
        graph.add_node("retrieve_links", self.log_wrapper("retrieve_links", self.retrieve_links))
        graph.add_node("check_rag_score", self.log_wrapper("check_rag_score", self.check_rag_score))
        graph.add_node("write_mail", self.log_wrapper("write_mail", self.write_mail))

        # Define main flow
        graph.set_entry_point("extract_jobs")
        graph.add_edge("extract_jobs", "check_coherence")

        # Conditional Edge: Retry if coherence_score < DEFAULT_COHERENCE_THRESHOLD
        graph.add_conditional_edges(
            "check_coherence",
            lambda state: "extract_jobs" if state["coherence_score"] < DEFAULT_COHERENCE_THRESHOLD else "retrieve_links"
        )

        graph.add_edge("retrieve_links", "check_rag_score")

        # Conditional Edge: Retry if rag_score < DEFAULT_RAG_SCORE_THRESHOLD
        graph.add_conditional_edges(
            "check_rag_score",
            lambda state: "retrieve_links" if state["rag_score"] < DEFAULT_RAG_SCORE_THRESHOLD else "write_mail"
        )

        return graph.compile()