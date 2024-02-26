import toml
import streamlit as st
from functions import *
import cohere
import re



parsed_toml = toml.load("secrets.toml")
api_key = parsed_toml['api_key']
co = cohere.Client(api_key)

model_path = "Palistha/GPT-2-finetuned-model-3"
tokenizer = "Palistha/GPT-2-finetuned-model-3"

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


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = load_data(query)
            context = ""
            for idx, r in enumerate(response):
                context += r.document['text']
                context += " "
            if not query.endswith("?"):
                query += "?"

            output_text = generate_answer(model_path,context,query)
            # print(output_text)
            match = re.search(r'[^?]*\?([^?]*)$', output_text)
            if match:
                ans = match.group(1).strip()
            source = get_urls(context)
            result_source = ', '.join(source) 
            output = ans + " "  + "\n Source:" + result_source
            try:
                if math.isnan(float(ans)):
                    output =  "Sorry, I could not generate a valid answer."
            except (ValueError, TypeError):
                output = ans + " "  + "\n Source:" + result_source 
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


