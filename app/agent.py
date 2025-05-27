# This file implements an agent-based approach using LangChain Agents,
# as described in the 'Agentic-Rag-LangChain-Agents' branch of the README.
# It is not used in the default LangGraph-based execution flow found in
# `main.py` and `chains.py`.

import os
import re # Added import re
import logging # Added import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from utils import clean_text
from portfolio import Portfolio
load_dotenv()

class AgenticChain:
    """
    Implements an agent-based workflow for generating cold emails using LangChain Agents.
    
    This class defines a series of tools (job scraping, data extraction, email generation)
    and initializes a LangChain agent to orchestrate these tools based on a query.
    This approach is an alternative to the LangGraph-based implementation in `chains.py`.

    Attributes:
        llm (ChatGroq): Instance of the Groq LLM used by the agent and tools.
        portfolio (Portfolio): Instance of the Portfolio class, used by the extract_job_details tool.
                               Note: The `links` attribute on the class instance is used to pass
                               data between `extract_job_details` and `generate_email` in a
                               previous implementation, which is now handled by combined string passing.
        scrape_tool (Tool): Tool for scraping job data from a URL.
        extract_tool (Tool): Tool for extracting structured job details and portfolio links.
        email_tool (Tool): Tool for generating a cold email from job details and links.
        agent (AgentExecutor): The initialized LangChain agent.
    """
    def __init__(self):
        """
        Initializes the AgenticChain, setting up the LLM, tools, and the agent.
        The LLM model name is hardcoded here to "llama-3.3-70b-specdec".
        """
        self.links = None # This attribute was used in a previous design for link passing.
                          # It's still set by extract_job_details but generate_email now
                          # receives links via its direct input string.
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-specdec" # Specific model for this agent approach
        )
        self.portfolio = Portfolio() # Initialize portfolio for use in extract_job_details

        # --- Tool Definitions ---

        def scrape_job(url: str) -> str:
            """
            Scrapes textual content from a job posting URL.

            Args:
                url (str): The URL of the job posting page.

            Returns:
                str: The cleaned textual content of the page.
                     Returns an empty string if loading or cleaning fails.
            """
            logging.info(f"Scraping job from URL: {url}")
            loader = WebBaseLoader([url])
            try:
                docs = loader.load()
                if not docs:
                    logging.warning(f"No documents loaded from URL: {url}")
                    return ""
                page_data = clean_text(docs[0].page_content)
            except Exception as e: # Broad exception for any loading/cleaning errors
                logging.error(f"Error scraping URL {url}: {e}")
                return "" # Return empty string on error
            # logging.debug(f"Scraped page data: {page_data[:200]}...") # Avoid logging full page
            return page_data
        
        self.scrape_tool = Tool(
            name="Job Scraper",
            func=scrape_job,
            description="Scrapes job descriptions and content from a given careers page URL. Input should be a single URL string."
        )

        # Prompt for Job Data Extractor Tool
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            very important Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        def extract_job_details(page_data: str) -> str:
            """
            Extracts structured job details from scraped page text using an LLM
            and retrieves relevant portfolio links.

            The extracted job details (as a JSON-like string) and the portfolio links
            (as a string representation of a list) are combined into a single
            output string, separated by markers.

            Args:
                page_data (str): The raw text content scraped from a job page.

            Returns:
                str: A combined string containing the extracted job details and
                     portfolio links, formatted with specific markers.
                     Example:
                     JOB_DETAILS_START
                     {'role': 'Engineer', 'skills': ['Python']}
                     JOB_DETAILS_END
                     LINKS_START
                     [{'links': 'http://example.com/project1'}]
                     LINKS_END
                     Returns an error string if extraction or link retrieval fails.
            """
            logging.info("Extracting job details and links...")
            chain_extract = prompt_extract | self.llm
            
            try:
                llm_response_content = chain_extract.invoke(input={'page_data': page_data}).content
                json_parser = JsonOutputParser()
                # 'res' here is the structured job details (typically a list of dicts, or a dict)
                extracted_jobs_data = json_parser.parse(llm_response_content)
                logging.debug(f"Extracted job details type: {type(extracted_jobs_data)}, content: {extracted_jobs_data}")

                # Assuming the first job item is the primary one if multiple are extracted
                skills = extracted_jobs_data[0].get('skills', []) if isinstance(extracted_jobs_data, list) and extracted_jobs_data else \
                         extracted_jobs_data.get('skills', []) if isinstance(extracted_jobs_data, dict) else []

                self.portfolio.load_portfolio() # Ensure portfolio is loaded
                # self.links will store the query_links result, e.g., [[{'links': 'url1'}], [{'links': 'url2'}]]
                self.links = self.portfolio.query_links(skills=skills) 
                logging.debug(f"Queried links: {self.links}")

            except OutputParserException as ope:
                logging.error(f"Error parsing LLM response in extract_job_details: {ope}")
                return "Error: Could not parse job details from LLM response."
            except Exception as e: # Catch other potential errors (e.g., during portfolio query)
                logging.error(f"Unexpected error in extract_job_details: {e}")
                return f"Error: An unexpected error occurred during job detail extraction: {e}"

            # Combine job details and links into a single string output for the next tool
            combined_output = f"""JOB_DETAILS_START
{res}
JOB_DETAILS_END
LINKS_START
{self.links}
LINKS_END"""
            return combined_output


        self.extract_tool = Tool(
            name="Job Data Extractor",
            func=extract_job_details,
            description="Extracts job details and relevant portfolio links from scraped page data. The input should be the raw text from a job page. Returns a single string combining job details and links, formatted with special markers."
        )

        # Define Cold Email Generator Tool
        def generate_email(combined_input_str: str) -> str:
            """
            Generates a personalized cold email using a combined string of job details and links.

            This function parses the `combined_input_str` to separate job details
            and portfolio links, then uses them to populate a prompt for an LLM
            to generate the email content.

            Args:
                combined_input_str (str): A string containing job details and links,
                    expected to be formatted with 'JOB_DETAILS_START/END' and
                    'LINKS_START/END' markers. Example:
                    JOB_DETAILS_START
                    {'role': 'Engineer', 'skills': ['Python']}
                    JOB_DETAILS_END
                    LINKS_START
                    [[{'links': 'http://example.com/project1'}]]
                    LINKS_END

            Returns:
                str: The generated cold email content. If parsing fails or an
                     error occurs during LLM invocation, an error message string is returned.
            """
            logging.info("Generating cold email from combined input string...")
            
            # Use re.DOTALL to make '.' match newlines as well
            job_details_match = re.search(r"JOB_DETAILS_START\s*(.*?)\s*JOB_DETAILS_END", combined_input_str, re.DOTALL)
            links_match = re.search(r"LINKS_START\s*(.*?)\s*LINKS_END", combined_input_str, re.DOTALL)

            if not job_details_match or not links_match:
                error_msg = "Error: Could not parse job_details and/or links from input. Ensure JOB_DETAILS_START/END and LINKS_START/END markers are present and correctly formatted."
                logging.error(error_msg + f" Input received: {combined_input_str[:500]}...") # Log part of the input
                return error_msg

            job_details_str = job_details_match.group(1).strip()
            links_str = links_match.group(1).strip()
            
            # The links_str is expected to be a string representation of a list of lists of dicts,
            # e.g., "[[{'links': 'url1'}], [{'links': 'url2'}]]".
            # The prompt needs to handle this format or it needs further processing here.
            # For now, we pass it as is, assuming the LLM or prompt can manage.
            # A more robust approach might involve ast.literal_eval and then joining links.
            # Example of further processing if needed:
            # try:
            #     import ast
            #     parsed_links_list_of_lists = ast.literal_eval(links_str)
            #     # Extract actual URLs:
            #     actual_links = [item['links'] for sublist in parsed_links_list_of_lists for item in sublist if 'links' in item]
            #     links_for_prompt = "\n".join(f"- {link}" for link in actual_links)
            # except (SyntaxError, ValueError) as e:
            #     logging.warning(f"Could not parse links_str '{links_str}' into a list: {e}")
            #     links_for_prompt = links_str # Fallback to raw string

            email_generation_prompt_template = """
            ### JOB DESCRIPTION:
            {job_details}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {links}
            Remember you are Mohan, BDE at AtliQ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            """
            )
            
            chain_email = prompt_email | self.llm
            # Pass parsed values
            response = chain_email.invoke(input={'job_details': job_details_str, 'links': links_str}) 
            return response.content

        self.email_tool = Tool(
            name="Cold Email Generator",
            func=generate_email,
            description="Generates a personalized cold email for recruiters. Input must be a single string combining job details and portfolio links, formatted with JOB_DETAILS_START/END and LINKS_START/END markers."
        )

        # Example of a refinement tool (currently commented out but documented for completeness)
        # def refine_email(existing_email: str, feedback: str) -> str:
        #     """
        #     Refines an existing cold email based on provided feedback using an LLM.
        #     (This tool is defined but not currently used in the active agent.)

        #     Args:
        #         existing_email (str): The email content that needs refinement.
        #         feedback (str): Feedback or instructions on how to refine the email.

        #     Returns:
        #         str: The refined email content.
        #     """
        # def refine_email(existing_email, feedback):
        #     """Refines an existing cold email based on recruiter feedback."""
        #     prompt_refine = PromptTemplate.from_template(
        #         """### EXISTING EMAIL:
        #         {existing_email}
        #         ### FEEDBACK:
        #         {feedback}
        #         ### INSTRUCTION:
        #         Refine the email to better match the feedback.
        #         ### REFINED EMAIL:
        #         """
        #     )
        #     chain_refine = prompt_refine | self.llm
        #     response = chain_refine.invoke(input={'existing_email': existing_email, 'feedback': feedback})
        #     return response.content

        # self.refine_tool = Tool(
        #     name="Email Refinement",
        #     func=refine_email,
        #     description="Refines a generated cold email based on recruiter feedback."
        # )

        # Initialize the Agent
        self.agent = initialize_agent(
            tools=[self.scrape_tool, self.extract_tool, self.email_tool], # List of tools agent can use
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # Standard agent type that decides actions based on tool descriptions
            verbose=True, # Enables detailed logging of the agent's thought process
            handle_parsing_errors=True # Gracefully handle errors if LLM output for tool use isn't parsable
        )

    def run_agent(self, query: str) -> str:
        """
        Runs the initialized LangChain agent with a given query.

        The agent will use the provided tools (scraper, extractor, email generator)
        to process the query and attempt to generate a cold email.

        Args:
            query (str): The input query for the agent, typically expected to be
                         a URL to a job posting for this application's context,
                         as the first step is usually scraping.

        Returns:
            str: The result from the agent's execution. This could be the
                 generated email, an error message, or intermediate output
                 depending on how the agent's execution concludes.
        """
        logging.info(f"Running agent with query: {query}")
        try:
            # The .run() method executes the agent and returns the final response.
            result = self.agent.run(query)
            logging.info(f"Agent Output: {result}")
        except Exception as e: # Catch any broad errors during agent execution
            logging.error(f"Exception during agent execution: {e}", exc_info=True)
            return f"Agent execution failed due to an error: {e}"

        # Simple check for "ERROR" string in result as a fallback, though specific
        # tool error handling should ideally return structured errors or use exceptions.
        if "ERROR" in str(result).upper(): 
            logging.error(f"Error string detected in agent execution output: {result}")
            # No specific action here, just logging; the result itself is returned.
        
        return result
