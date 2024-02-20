import streamlit as st
from streamlit_chat import message
import random
from functions import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import notebook_login
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, T5TokenizerFast


# notebook_login()


modelPath = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    encode_kwargs=encode_kwargs
)
save_directory = "vector_db"
database = FAISS.load_local(save_directory,embeddings)
model_path = "Palistha/finetuned-t5-small"
tokenizer = "Palistha/finetuned-t5-small"
max_len=1024

# Display chat messages from history on app rerun
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask something")
if query:
    with st.chat_message("user"):
        st.write(query)
    # message(prompt, is_user=True)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": query})


    @st.cache_resource(show_spinner=True)
    def load_data(query):
        search = database.similarity_search(query,k=1)
        for doc in search:
            context = doc.page_content
            source = doc.metadata['source']
        return context,source


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context,source = load_data(query)
            if not context.endswith("."):
                context += "."
            if not query.endswith("?"):
                query += "?"
            answer = generate_answer(model_path,context,query)
            try:
                if math.isnan(float(answer)):
                    output =  "Sorry, I could not generate a valid answer."
            except (ValueError, TypeError):
                output = answer + " "  + "\n Source:" + source
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


