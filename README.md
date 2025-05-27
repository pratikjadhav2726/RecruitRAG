
# RecruitRAG: Agentic email generator 

### AI-Powered Cold Email Generation using RAG, LangChain, and Llama 3  

---

## 📌 Project Overview  

This project is an **AI-powered cold email generator** built using **Retrieval-Augmented Generation (RAG), LangChain, and Llama 3**, designed to **automate personalized outreach and improve recruiter response rates**.  

### There are three different approaches, implemented on separate branches:
####  1.  Basic RAG with LangChain – Uses standard retrieval-augmented generation.
#### 	2.	Agentic RAG with LangChain Agents – Implements autonomous agent behavior for query understanding and email construction.
####  3.	Agentic RAG with LangGraph – Adds retry mechanisms using coherence and RAG score checks for enhanced precision.

[Placeholder for Architecture Diagram 1 - Formerly `![architecture](...)`]
[Placeholder for Architecture Diagram 2 - Formerly `<img .../>`]


### 🔥 Features:  
✅ **RAG-based AI model**: Enhances contextual relevance in email generation.  
✅ **ChromaDB Integration**: Efficient **vector-based candidate-job matching** for skill-based recommendations.  
✅ **Streamlit UI**: Recruiters can generate tailored cold emails **in under 30 seconds**, boosting efficiency by **40%**.  
✅ **Enhanced Engagement**: Personalized AI-generated emails improve recruiter response rates **by 30%**.  

### MCP Server Integration: 
We plan to integrate Model Context Protocol (MCP) server capabilities to enhance RecruitRAG’s functionality. This will enable seamless integration with external data sources, third-party services, messaging platforms, scheduling tools, and advanced analytics. These additions will empower dynamic, context-aware interactions across multiple channels, improving both efficiency and user experience.

### Branches and Approaches

| Branch                          | Approach                          | Key Files                               | Key Features                                                                            |
|---------------------------------|-----------------------------------|-----------------------------------------|-----------------------------------------------------------------------------------------|
| `Cold-email-generator`          | Basic RAG with LangChain          | (N/A for main branch focus)             | Simple RAG-based retrieval and email generation.                                        |
| `Agentic-Rag-LangChain-Agents`  | Agentic RAG with LangChain Agents | `app/agent.py` (alternative approach)   | Uses agents for dynamic tool selection. (Not the primary approach on main branch)       |
| `Agentic-Rag_LangGraph`         | Agentic RAG with LangGraph        | `app/chains.py`, `app/main.py`          | **Primary Recommended Approach.** Adds coherence and RAG score validation with retries. |

**Note:** The `Agentic-Rag_LangGraph` approach, primarily implemented in `app/chains.py` and executed via `app/main.py`, is the most feature-complete and recommended version on the `main` branch. Other branches represent alternative or earlier developmental approaches.

#### 1️⃣ Basic RAG with LangChain (Cold-email-generator branch)
	•	Uses Retrieval-Augmented Generation (RAG) for contextual relevance.
	•	Basic retrieval and email generation with no agent involvement.

#### 2️⃣ Agentic RAG with LangChain Agents (Agentic-Rag-LangChain-Agents branch)
	•	Introduces LangChain agents for query understanding and multi-step execution (`app/agent.py`).
	•	Automates data retrieval, filtering, and email generation dynamically.

#### 3️⃣ Agentic RAG with LangGraph (Agentic-Rag_LangGraph branch)
	•	Implements LangGraph for structured execution (primarily in `app/chains.py` and `app/main.py`).
	•	Coherence and RAG Score Validation:
	•	If coherence < 0.8, the job extraction step retries.
	•	If RAG score < 0.8, retrieval step retries before generating the email.


---

## 🏗️ Tech Stack  

- **📌 Llama 3** – AI-powered natural language generation  
- **📌 LangChain** – Framework for RAG-based AI workflows  
- **📌 ChromaDB** – Vector database for candidate-job matching  
- **📌 Streamlit** – User-friendly web UI  
- **📌 Python** – Backend logic and model integration
- **📌 LangSmith** – Tracebility and Observability

---

## 🎯 How It Works  

1. **User Input**: Recruiters enter details about the candidate, job role, and desired personalization.  
2. **Data Retrieval**: ChromaDB fetches relevant job-candidate matches using **vector embeddings**.  
3. **RAG-powered Generation**: LangChain + Llama 3 generate **highly personalized** cold emails.
4. **Agentic RAG and LangGraph Versions**:
	•	LangChain Agents dynamically route and process retrieval.
	•	LangGraph ensures adaptive execution, retrying steps if coherence or RAG scores are low. 
5. **UI Display**: The generated email is displayed in **Streamlit**, ready to be copied and sent.

[Placeholder for UI Example Image - Formerly `<img .../>`]
---

## 📥 Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/cold-email-generator-rag.git
cd cold-email-generator-rag
```
### Checkout a Specific Branch

### For basic RAG:

```bash 
git checkout Cold-email-generator
```

### For Agentic RAG with LangChain Agents:

```bash 
git checkout Agentic-Rag-LangChain-Agents
```

### For Agentic RAG with LangGraph:

```bash 
git checkout Agentic-Rag_LangGraph
```

### Python Version
It is recommended to use **Python 3.10 or higher** for this project. Ensure your environment meets this requirement before proceeding.

### Create and Activate a Virtual Environment

```bash
python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate  
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Environment Variables
Before running the application, create a `.env` file in the project root directory. This file is automatically loaded by the application modules (`app/chains.py` and `app/agent.py`) using `python-dotenv` when the application starts or when these modules are first imported.

The following variables are supported:

*   **`GROQ_API_KEY` (Required)**: Your API key for accessing Groq cloud services. This is essential for the LLM (Large Language Model) to function.
*   **`GROQ_LLM_MODEL_NAME` (Optional)**: Specifies the Groq LLM model to be used by the LangGraph approach (`app/chains.py`). 
    *   Defaults to `llama-3.3-70b-versatile` if not set (this default is defined in `app/chains.py`).
    *   The `app/agent.py` (alternative approach) uses a hardcoded model (`llama-3.3-70b-specdec`) which is not affected by this environment variable.
    *   Example values: `mixtral-8x7b-32768`, `gemma-7b-it`.

Example `.env` file:
```env
GROQ_API_KEY="your_actual_groq_api_key_here"
# GROQ_LLM_MODEL_NAME="mixtral-8x7b-32768" # Uncomment to override default for app/chains.py
```

### Configuration Constants
Key operational parameters for the primary LangGraph approach (defined in `app/chains.py`), such as RAG and coherence score thresholds, as well as the sender's persona details (name, company, description), are defined as constants directly at the top of the `app/chains.py` file. These can be modified in the code to tweak application behavior (e.g., `DEFAULT_COHERENCE_THRESHOLD`, `SENDER_PERSONA_NAME`).

### Vector Store for Portfolio
The application uses ChromaDB to store and query portfolio project embeddings. These embeddings are generated from the `app/resource/my_portfolio.csv` file. ChromaDB persists these embeddings in the `vectorstore/` directory, which will be created automatically in the project root during the first run when `Portfolio().load_portfolio()` is called (typically on application startup or when portfolio data is first accessed). 

**Important:**
*   This `vectorstore/` directory is essential for the portfolio feature to work across sessions.
*   If you modify the `my_portfolio.csv` file, you may need to **delete the `vectorstore/` directory** to force the application to re-index the data and reflect your changes.

---

## ▶️ Running the App  

Launch the **Streamlit** UI:  

```bash
streamlit run app/main.py
```

The application will be accessible at `http://localhost:8501/`.  
 

---

## 📊 Example Output  

```plaintext
Subject: Expert Machine Learning Solutions for Your Business

Dear Hiring Manager,

I came across the job description for a Senior Machine Learning Engineer at your organization and was impressed by the scope of the role. As a Business Development Executive at AtliQ, I believe our team can provide the expertise and support you need to develop and implement integrated software algorithms that structure, analyze, and leverage data in product and systems applications.

At AtliQ, we have a proven track record of empowering enterprises with tailored solutions that foster scalability, process optimization, cost reduction, and heightened overall efficiency. Our team of experts has extensive experience in machine learning, data science, and software engineering, with a strong focus on developing and communicating descriptive, diagnostic, predictive, and prescriptive insights.

Our capabilities align perfectly with your requirements, including:

* Machine learning algorithms and data science methods
* Data wrangling, feature engineering, and time series forecasting
* Natural language processing, computer vision, and deep learning methods
* Solution design, technical design, and hyperparameter tuning
* Cloud technologies, including Google Cloud, AWS, and distributed systems

I'd like to highlight some of our relevant portfolio work:
* https://example.com/ml-python-portfolio (Machine Learning and Python solutions)
* https://example.com/java-portfolio (Java-based solutions)
* https://example.com/ml-python-portfolio (Machine Learning and Python solutions for data science and analytics)

Our team is well-versed in a range of technologies, including Java, Spark, Scala, Python, R, SAS, SQL, and more. We're confident that our expertise can help you develop and evaluate algorithms to improve product/system performance, quality, data management, and accuracy.

I'd love to schedule a call to discuss how AtliQ can support your machine learning initiatives and help you achieve your business goals. Please let me know if you're interested, and we can schedule a time that suits you.

Best regards,

Mohan
Business Development Executive
AtliQ 
```

---

## 🛠️ To-Do  

- [ ] Add **GPT-4 fine-tuning** for more refined personalization  
- [ ] Improve **candidate-job matching algorithms**  
- [ ] Optimize **email sentiment analysis**

---

##  troubleshooting

Here are some common issues and their solutions:

*   **API Key Issues:**
    *   **Symptom:** Authentication errors, messages about missing API keys, or HTTP 401/403 errors from Groq when the application tries to use the LLM.
    *   **Solution:** Ensure your `GROQ_API_KEY` is correctly set in your `.env` file located in the project root directory. Verify that the key is valid, has not expired, and has the necessary permissions/credits for the Groq services. The application modules (`app/chains.py`, `app/agent.py`) use `python-dotenv` to load this file automatically.

*   **Python Version/Dependency Conflicts:**
    *   **Symptom:** Errors during `pip install -r requirements.txt` (e.g., packages cannot be found or build errors), `ModuleNotFoundError` at runtime for installed packages, or unexpected behavior from specific libraries.
    *   **Solution:** 
        *   Confirm you are using Python 3.10 or higher. You can check with `python --version`.
        *   It's highly recommended to use a virtual environment (like `venv` or `conda`) to manage project dependencies and avoid conflicts with system-wide packages. Ensure your virtual environment is activated when running `pip install` and `streamlit run`.
        *   If specific package errors occur, try reinstalling the problematic package (e.g., `pip uninstall <package_name>` then `pip install <package_name>`), checking its documentation for known compatibility issues, or creating a fresh virtual environment and reinstalling all requirements.

*   **ChromaDB Issues / Portfolio Not Working:**
    *   **Symptom:** Portfolio links are not appearing in generated emails, errors related to "vectorstore" during application startup or when querying skills, or the application seems to "forget" portfolio items between runs (if persistence is not working).
    *   **Solution:**
        *   Ensure the `app/resource/my_portfolio.csv` file exists, is correctly formatted (as a CSV with "Techstack" and "Links" columns), and contains valid data.
        *   The `vectorstore/` directory should be automatically created by ChromaDB in the project root. Check if it exists. If not, the application might not have write permissions to the project root directory to create it.
        *   If you've updated `my_portfolio.csv` or suspect an issue with the stored embeddings (e.g., outdated data), you can safely **delete the entire `vectorstore/` directory**. The application will attempt to rebuild it from the CSV when `Portfolio().load_portfolio()` is next triggered (usually on the first operation that requires portfolio data).

*   **No Emails Generated / Issues with LLM Output:**
    *   **Symptom:** The application runs without critical errors but produces no emails, or the generated emails are nonsensical, incomplete, or do not seem to relate well to the job description.
    *   **Solution:**
        *   Verify the job URL provided is valid, public, and leads to a page with a clear job description. Some career pages might use complex JavaScript that makes scraping difficult, or the job content might be too sparse.
        *   The scraped content might be minimal after the `clean_text` process, leaving little for the LLM to work with. You can add temporary `logging.debug` statements in `app/main.py` to inspect `scraped_text`.
        *   Check the application logs (console output where `streamlit run app/main.py` is executed) for any errors or warnings from the LLM, parsing stages, or during the LangGraph execution. `app/chains.py` and `app/agent.py` include logging.
        *   If you are using a custom `GROQ_LLM_MODEL_NAME` (via the `.env` file), ensure it's a valid model identifier for the Groq API and that it's suitable for the kind of generation tasks required by the prompts in `app/chains.py`. Some models might be better suited for chat versus text completion or instruction following.
        *   Review the "Configuration Constants" in `app/chains.py` (e.g., `DEFAULT_COHERENCE_THRESHOLD`, `DEFAULT_RAG_SCORE_THRESHOLD`). If these thresholds are too strict for the quality of job data being extracted, valid jobs might be filtered out before the email generation stage. Try temporarily lowering them for testing.

---

## 🤝 Contribution  

Contributions are welcome! Feel free to **fork** the repo, create a **new branch**, and submit a **pull request**.  

---

## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 📧 Contact  

💬 **Have questions? Let's connect!**  
📧 Email: [PratikJadhav2726@gmail.com]
🔗 GitHub: [(https://github.com/pratikjadhav2726)]  
🔗 LinkedIn: (https://www.linkedin.com/in/pratikjadhav2726)

---

⭐ **If you like this project, don't forget to give it a star!** ⭐
