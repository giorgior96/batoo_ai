import streamlit as st
from langchain_helper import execute_user_query
import pysqlite3
import sqlite3


st.title("About Ana ✨")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
with st.chat_message("assistant"):
    st.write("Hello, you can ask me anything about Ana 👋")
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query_text := st.chat_input("Ask away!"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query_text)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query_text})

if query_text:
    response = execute_user_query(query_text)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})