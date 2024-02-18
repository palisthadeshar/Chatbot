import streamlit as st
from streamlit_chat import message
import random
from functions import *
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import notebook_login
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# notebook_login()


modelPath = "sentence-transformers/all-mpnet-base-v2"
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    encode_kwargs=encode_kwargs
)
save_directory = "vector_db"
database = FAISS.load_local(save_directory,embeddings)
model_path = "Palistha/finetuned-gpt2"
tokenizer="Palistha/finetuned-gpt2"
max_len=1024
# query = "Who is the CEO of TAI?"

    

#output
input_text = '''Mission and Vision | TAI 日本語 Contact OUR STORY ABOUT US Introduction Message from CEO Team Service About Nepal SDGs BLOG English 日本語 Contact About TAI Vision Smiling Nepal TAI was established with the vision of “Smiling Nepal”. TAI is dedicated in assisting hardworking and versatile Nepali IT graduates in contributing to their country via Technology and Intelligence. We are looking forth to becoming an IT bridge between Nepal and Japan. Mission Accelerate the GDP of Nepal by 1%. We want to create highly skilled jobs in our own country and stop brain-drain of the nation. Our motto is to "Serve Nation With Innovation". Our ultimate goal is to become an offshore company large enough to accelerate the GDP of Nepal by 1%. Company Info Company Name:TAI Inc (Technology and Intelligence) Established:Nepal (2020), Japan (2022) CEO:Sharad Rai Main Bank: Mitsubishi UFJ Bank PayPay Bank Capital:1 million yen (30 million yen including Nepal Group) Employees:70 (Including Nepal and Japan Group employees) Address: Matsumoto Building-502, 10 Iwato-cho, Shinjuku-ku, Tokyo, Japan Kathmandu, Shantinagar, Nepal Company Services Contact Blog Services Artificial Intelligence Digital Transformation Cloud & DevOps Web Application Development Mobile Application Development UI/UX Design Connect With Us Address Matsumoto Building-502, 10 Iwato-cho, Shinjuku-ku, Tokyo, Japan SanoKharibot Shantinagar, Kathmandu, Nepal Copyright 2024 TAI Inc. All rights reserved. Privacy Policy Terms of Use ''' + "What is the mission of TAI?"
# # input_text = context + " " + query
# model_path = "Palistha/finetuned-gpt2"
# tokenizer="Palistha/finetuned-gpt2"
# seq = input_text
# max_len=1024
# output_text = generate_text(model_path,tokenizer,seq,max_len)
# parts = output_text.split(input_text)
# answer = parts[-1].strip()
# print("Answer:", answer)




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
    def load_data():
        search = database.similarity_search(query,k=1)
        print(search)
        for doc in search:
            context = doc.page_content
            source = doc.metadata['source']
        return context,source


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context,source = load_data()
            input_text = context + " " + query
            output_text = generate_text(model_path,tokenizer,input_text,max_len)
            parts = output_text.split(input_text)
            answer = parts[-1].strip()
            output = answer + " "  + "\n Source:" + source
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


