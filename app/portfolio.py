import pandas as pd
import chromadb
import uuid


class Portfolio:
    """
    Manages a portfolio of projects stored in a CSV file and provides
    functionality to load and query this data using a ChromaDB vector store.

    Attributes:
        DEFAULT_QUERY_N_RESULTS (int): Default number of results to return from queries.
        file_path (str): Path to the CSV file containing portfolio data.
        data (pd.DataFrame): Pandas DataFrame holding the loaded portfolio data.
        chroma_client (chromadb.PersistentClient): ChromaDB client instance.
        collection (chromadb.Collection): ChromaDB collection for portfolio items.
    """
    DEFAULT_QUERY_N_RESULTS = 2  # Default number of results for portfolio link queries

    def __init__(self, file_path: str = "app/resource/my_portfolio.csv"):
        """
        Initializes the Portfolio object, loads data from the CSV, and sets up
        the ChromaDB client and collection.

        Args:
            file_path (str, optional): The path to the CSV file containing
                portfolio data. Defaults to "app/resource/my_portfolio.csv".
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        # Initialize ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(path='vectorstore') # Specify path for clarity
        # Get or create the collection for portfolio items
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """
        Loads portfolio data from the CSV into the ChromaDB collection.

        This method currently only adds documents to the collection if the collection
        is empty. It does not handle re-indexing or updates if the underlying CSV
        data changes after the initial load.

        To re-index if the CSV changes, the 'vectorstore' directory (or the specific
        directory used by PersistentClient for this collection) may need to be 
        manually deleted before re-running this method, or a dedicated re-indexing 
        script/function could be implemented.
        """
        # Only load data if the collection is currently empty to avoid duplicates
        if not self.collection.count():
            for _, row in self.data.iterrows():
                # Add each project's tech stack as a document, with its link as metadata.
                # A unique ID is generated for each entry.
                self.collection.add(
                    documents=row["Techstack"],  # The text content to be vectorized and indexed
                    metadatas={"links": row["Links"]},  # Associated metadata (the project link)
                    ids=[str(uuid.uuid4())]  # Unique ID for the document
                )

    def query_links(self, skills: list[str], n_results: int = None) -> list[dict]:
        """
        Queries the ChromaDB collection for portfolio links relevant to the given skills.

        Args:
            skills (list[str]): A list of skill strings to query for. 
                                For example: ["Python", "Data Analysis"].
            n_results (int, optional): The number of top matching results to return. 
                                       Defaults to `self.DEFAULT_QUERY_N_RESULTS`.

        Returns:
            list[dict]: A list of metadata dictionaries from the top matching documents. 
                        Each dictionary is expected to contain a "links" key with the 
                        URL of the portfolio item. For example: 
                        `[[{'links': 'http://example.com/project1'}], [{'links': 'http://example.com/project2'}]]`
                        Returns an empty list if no relevant documents are found or if skills list is empty.
        """
        if not skills: # Handle empty skills list
            return []
        
        if n_results is None:
            n_results = self.DEFAULT_QUERY_N_RESULTS
        
        # Perform the query against the collection
        query_results = self.collection.query(
            query_texts=skills,  # List of texts to find similar documents for
            n_results=n_results  # Number of results to retrieve
        )
        # Extract and return the 'metadatas' field, which contains the links
        # The structure of query_results can be complex, e.g., {'ids': [[]], 'distances': [[]], 'metadatas': [[{'links': '...'}]], ...}
        # We are interested in the list of metadatas.
        return query_results.get('metadatas', [])
