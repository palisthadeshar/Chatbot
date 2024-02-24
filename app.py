import toml
import streamlit as st
from streamlit_chat import message
import random
from functions import *
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import cohere



parsed_toml = toml.load("secrets.toml")
api_key = parsed_toml['api_key']
co = cohere.Client(api_key)

# modelPath = "sentence-transformers/all-mpnet-base-v2"
# encode_kwargs = {'normalize_embeddings': False}
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,
#     encode_kwargs=encode_kwargs
# )
# save_directory = "vector_db"
# database = FAISS.load_local(save_directory,embeddings)

# model_path = "Palistha/finetuned-t5-small"
# tokenizer = "Palistha/finetuned-t5-small"

# model_path = "Palistha/finetuned-gpt2"
# tokenizer = "Palistha/finetuned-gpt2"

model_path = "Palistha/GPT-2-finetuned-model"
tokenizer = "Palistha/GPT-2-finetuned-model"
# max_len=1024
with open("data/file.txt", 'r') as file:
    lst = file.readlines()
    data = [item.strip() for item in lst]


st.subheader("Ask About TAI.")
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
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": query})


    @st.cache_resource(show_spinner=True)
    def load_data(query):
        response = co.rerank(
            model = 'rerank-english-v2.0',
            query = query,
            documents = data,
            top_n = 4
            )
        return response
        # search = database.similarity_search(query,k=4)
        # context = ""
        # source = []
        # for doc in search:
        #     text = doc.page_content
        #     context += text
        #     link = doc.metadata['source']
        #     source.append(link)
        # return context,source


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = load_data(query)
            context = ""
            for idx, r in enumerate(response):
                context += r.document['text']
                context += " "
            # if not context.endswith("."):
            #     context += "."
            if not query.endswith("?"):
                query += "?"

            output_text = generate_answer(model_path,context,query)
            # import pdb; pdb.set_trace()
            input_texts = context + " " + query
            parts = output_text.split(input_texts)
            ans = parts[-1].strip()
            source = get_urls(context)
            output = ans + " "  + "\n Source:" + source
            try:
                if math.isnan(float(ans)):
                    output =  "Sorry, I could not generate a valid answer."
            except (ValueError, TypeError):
                output = ans + " "  + "\n Source:" + source 
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


