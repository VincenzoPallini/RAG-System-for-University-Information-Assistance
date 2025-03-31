## Project Description: RAG System for University Information Assistance

### Project Goal

This project implements a Retrieval-Augmented Generation (RAG) system designed to address the challenges in accessing university information, which is often fragmented and difficult to find. The goal is to provide users with quick and simplified access to university documents (such as guides, regulations, etc.) by generating contextualized and precise answers to their questions. The system aims to improve the interactivity and accessibility of information compared to traditional methods.

### Architecture and Functionality

The system follows a standard RAG architecture:

1.  **Database Population:** PDF documents related to the university are loaded, split into smaller text blocks (chunks), and processed to generate vector embeddings. These embeddings are stored in a ChromaDB vector database. The `populate_db.py` script manages this process, ensuring the database is updated efficiently.
2.  **Retrieval:** When a user asks a question via the graphical user interface, the system performs a similarity search in ChromaDB to retrieve the text chunks most relevant to the query.
3.  **Generation:** The retrieved chunks are used as additional context for a Large Language Model (LLM). The original question and the retrieved context are formatted into a prompt sent to the LLM (e.g., Ollama with the `phi4:14b` model), which generates an answer based on this information.
4.  **User Response:** The LLM-generated response, along with references to the source documents, is presented to the user in the chat interface.
5.  **Orchestration:** The LangChain framework is used to orchestrate the data flow and interactions between the different components (loader, text splitter, vector database, LLM).

### Evaluation and Results

An evaluation system (`evaluate.py`) has been implemented using the CRAG (Corrective RAG) methodology and the RAGElo framework to measure the system's performance.

1.  **Dataset Generation:** Questions and answers (ground truth) are automatically generated from document chunks to create an internal evaluation dataset.
2.  **Metrics:** Performance is evaluated using metrics such as Quality, Trustworthiness, Originality, and a fact-based score (Fact Score) calculated via semantic similarity (SentenceTransformer) and LLM-based evaluation with RAGElo. Aggregate metrics like overall accuracy, hallucination rate, and coverage are also calculated.
3.  **Results:**
    * A significant difference in accuracy was observed between the LLM models tested for generation: Phi-4 (14b) achieved 85.86% accuracy, significantly outperforming Gemma 2 (2b) at 55.21%.
    * A qualitative comparison with ChatGPT on a specific question regarding university fees showed that the custom RAG system was able to provide a detailed answer, calculated based on the specific data from the provided documents and citing sources, whereas ChatGPT provided a more generic and potentially inaccurate answer for the specific case. Detailed evaluation results (accuracy, hallucination rate, coverage) and error analysis are logged and saved to an `evaluation_output.txt` file.

### Technologies Used

* **Language:** Python
* **Core AI/ML:**
    * **LangChain:** Framework for orchestrating LLM applications.
    * **Ollama:** For running LLMs locally (Models used: `phi4:14b` for generation/chat and evaluation, `gemma2:2b` for QA generation in testing, `nomic-embed-text` for embeddings).
    * **ChromaDB:** Vector database for indexing and retrieving embeddings.
    * **SentenceTransformers:** Library for calculating semantic similarity (Model used: `all-MiniLM-L6-v2`).
    * **RAGElo:** Framework for automated evaluation of RAG systems.
* **Data Processing:** `langchain_text_splitters`, `langchain_community.document_loaders` (`PyPDFDirectoryLoader`).
* **User Interface:** Tkinter.
* **Other Libraries:** `argparse`, `os`, `shutil`, `threading`, `numpy`, `json`, `re`, `logging`, `tqdm`, `asyncio`.
