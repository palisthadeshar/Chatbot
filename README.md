## Chatbot

### Overview

The Chatbot is developed using FAISS Similarity Search and Fine-tuned T5 Model. It is an advanced conversational system designed to provide accurate responses by combining two powerful techniques: similarity search with FAISS and context-based answer generation using a fine-tuned T5 model.

### Key Features:
* FAISS Integration: Utilizes FAISS (Facebook AI Similarity Search) for efficient similarity search within a large corpus of dataset that was scraped from target website.

* Document Similarity Search: Retrieves relevant documents based on the semantic similarity with the user query using FAISS.

* Fine-tuned T5 Model: Employs a fine-tuned T5 (Text-To-Text Transfer Transformer) model for contextual answer generation based on the context of the retrieved documents.

* Contextual Answer Generation: The chatbot analyzes the retrieved documents to generate a coherent and contextually relevant answer to the user query.

* Response: The generated answer is presented to the user via the streamlit interface.


### Technologies Used
* FAISS: Efficient similarity search library developed by Facebook AI Research.

* Python: Programming language used for implementing the chatbot logic, fine-tuning model and integration with FAISS.

* Streamlit: Web application framework used for building the chatbot interface.

* Natural Language Processing (NLP) Libraries: Libraries such as NLTK, Transformers for natural language understanding and processing.


### How it Works
* User Query: The user submits a query or question to the chatbot.

* Similarity Search: The chatbot leverages FAISS to perform similarity searches within the document corpus, identifying documents semantically similar to the user query.

* Document Retrieval: Relevant documents are retrieved based on the similarity scores obtained from FAISS.

* Fine-tuned T5 Model: The retrieved documents serve as context inputs to a fine-tuned T5 model.

* Answer Generation: The T5 model generates contextual answers based on the information extracted from the retrieved documents.

* Response: The synthesized answer is presented to the user via the streamlit application's interface. 



### Installation

1. Clone this repository to your local machine:

```git clone <repository_url>```


2. Navigate to the project directory:

```cd <project-directory>```

3. Create a virtual environment using venv:

```python -m venv venv```

4. Activate virtual environment:

On Windows: ```venv\Scripts\activate```

On macOS and Linux: ```source venv/bin/activate```


5. Install dependencies from requirements.txt:

```pip install -r requirements.txt```


### Usage

1. Ensure your virtual environment is activated.

2. Run the Streamlit application.

```streamlit run app.py```

3. Open a web browser and go to http://localhost:8501 to view the application.


### File Structure
* app.py: Main script containing the Streamlit application code.

* functions.py: Script that contains all the functions used in the project.

* requirements.txt: File listing all Python dependencies required for the project.

* README.md: Documentation file providing information about the project.