import os
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
    def __init__(self):
        self.links=None
        self.llm = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.3-70b-specdec"
        )

        # Define Job Scraper Tool
        def scrape_job(url):
            """Scrapes job data from a given URL."""
            loader = WebBaseLoader(url)
            page_data = clean_text(loader.load().pop().page_content)
            # print("hi",page_data)
            return page_data
        
        self.scrape_tool = Tool(
            name="Job Scraper",
            func=scrape_job,
            description="Scrape job descriptions from a given careers page URL."
        )

        # Define Job Data Extractor Tool
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

        def extract_job_details(page_data):
            """Extracts job details from scraped job descriptions."""
            chain_extract = prompt_extract | self.llm
            res = chain_extract.invoke(input={'page_data': page_data})
            
            # try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
            print("hi",type(res),res)
            skills = res[0].get('skills', [])
            porfolio = Portfolio()
            porfolio.load_portfolio()
            self.links=porfolio.query_links(skills=skills)
            print(self.links)
            # print(links)
            # except OutputParserException:
            #     raise OutputParserException("Context too big. Unable to parse jobs.")
            return res


        self.extract_tool = Tool(
            name="Job Data Extractor",
            func=extract_job_details,
            description="Extracts job details  from scraped page data from job scapper tool."
        )

        # Define Cold Email Generator Tool
        def generate_email(job_details):
            """Generates a personalized cold email for the given job description."""
            prompt_email = PromptTemplate.from_template(
                """
            ### JOB DESCRIPTION:
            {job_details}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {self.links}
            Remember you are Mohan, BDE at AtliQ. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
            )
            # print(job_details,links)
            
            chain_email = prompt_email | self.llm
            response = chain_email.invoke(input={'job_details': job_details})
            return response.content

        self.email_tool = Tool(
            name="Cold Email Generator",
            func=generate_email,
            description="Generates a personalized cold email for recruiters from job details and links."
        )

        # # Define Email Refinement Tool
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
            tools=[self.scrape_tool, self.extract_tool, self.email_tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Lets model dynamically choose tools
            verbose=True
        )

    def run_agent(self, query): 
        """Runs the agent with a given query and ensures correct data flow."""
        result = self.agent.run(query)

        # Debug: Print output at each stage
        print(f"🔹 Agent Output: {result}")

        if "ERROR" in result:
            print("🚨 Error detected in job scraping. Returning error message.")
            return result  # Return error if scraping failed

        return result

