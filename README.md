
# RecruitRAG: Agentic email generator 

### AI-Powered Cold Email Generation using RAG, LangChain, and Llama 3  

---

## 📌 Project Overview  

This project is an **AI-powered cold email generator** built using **Retrieval-Augmented Generation (RAG), LangChain, and Llama 3**, designed to **automate personalized outreach and improve recruiter response rates**.  

### There are three different approaches, implemented on separate branches:
####  1.  Basic RAG with LangChain – Uses standard retrieval-augmented generation.
#### 	2.	Agentic RAG with LangChain Agents – Implements autonomous agent behavior for query understanding and email construction.
####  3.	Agentic RAG with LangGraph – Adds retry mechanisms using coherence and RAG score checks for enhanced precision.

![architecture](https://github.com/user-attachments/assets/6bfd4160-8d27-4b9b-8c10-745460e9f25c)
<img width="800" alt="image" src="https://github.com/user-attachments/assets/6196daca-c52c-4dc1-99bc-c0ef25e88e2c" style="float: left; margin-right: 15px;"/>



### 🔥 Features:  
✅ **RAG-based AI model**: Enhances contextual relevance in email generation.  
✅ **ChromaDB Integration**: Efficient **vector-based candidate-job matching** for skill-based recommendations.  
✅ **Streamlit UI**: Recruiters can generate tailored cold emails **in under 30 seconds**, boosting efficiency by **40%**.  
✅ **Enhanced Engagement**: Personalized AI-generated emails improve recruiter response rates **by 30%**.  

### MCP Server Integration: 
We plan to integrate Model Context Protocol (MCP) server capabilities to enhance RecruitRAG’s functionality. This will enable seamless integration with external data sources, third-party services, messaging platforms, scheduling tools, and advanced analytics. These additions will empower dynamic, context-aware interactions across multiple channels, improving both efficiency and user experience.

### Branches and Approaches

| Branch                          | Approach                          | Key Features                                                                            |
|---------------------------------|-----------------------------------|-----------------------------------------------------------------------------------------|
| `Cold-email-generator`          | Basic RAG with LangChain          | Simple RAG-based retrieval and email generation.                                        |
| `Agentic-Rag-LangChain-Agents`  | Agentic RAG with LangChain Agents | Uses agents for dynamic tool selection. |
| `Agentic-Rag_LangGraph`         | Agentic RAG with LangGraph        | Adds coherence and RAG score validation with retries for improved accuracy.             |

#### 1️⃣ Basic RAG with LangChain (Cold-email-generator branch)
	•	Uses Retrieval-Augmented Generation (RAG) for contextual relevance.
	•	Basic retrieval and email generation with no agent involvement.

#### 2️⃣ Agentic RAG with LangChain Agents (Agentic-Rag-LangChain-Agents branch)
	•	Introduces LangChain agents for query understanding and multi-step execution.
	•	Automates data retrieval, filtering, and email generation dynamically.

#### 3️⃣ Agentic RAG with LangGraph (Agentic-Rag_LangGraph branch)
	•	Implements LangGraph for structured execution.
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

<img width="370" alt="image" src="https://github.com/user-attachments/assets/2ae71fdb-b133-4108-aadd-8475e1410887" style="float: left; margin-right: 15px;" />
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

### Create and Activate a Virtual Environment

```bash
python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate  
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

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
