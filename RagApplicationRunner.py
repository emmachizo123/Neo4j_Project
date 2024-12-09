"""
file holds our frontend with Streamlit
"""
from typing import Set


from RagApplicationTemp import lookup

import streamlit as st
from streamlit_chat import message


#Step 1
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

#Step 2
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages =True
)

MessagesPlaceholder(variable_name="chat_history")# where memory is stored

st.header("ZH Technologies RAG Application ")
prompt = st.text_input("Prompt", placeholder = "Ask me your question?..")

# session persistence
#persist user prompt history. First iteration set it to an empty list

if(
    "user_prompt_history" not in st.session_state
    and "chat_answers_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["user_prompt_history"] =[]
    st.session_state["chat_answers_history"] =[]
    st.session_state["chat_history"] = []


if prompt:

    with st.spinner("Generating Response"):

        generated_response = lookup(question=prompt,chat_history=st.session_state["chat_history"])
        #generated_response =lookup(question=prompt, chat_history=st.session_state["chat_history"])


        st.session_state["user_prompt_history"].append(prompt)
        #st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_answers_history"].append(dict(generated_response)["output"])
        st.session_state["chat_history"].append(("human",prompt))
        st.session_state["chat_history"].append(("ai", dict(generated_response)["output"]))

if st.session_state["chat_answers_history"]:
    for generated_response,user_query in zip( st.session_state["chat_answers_history"],st.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)




