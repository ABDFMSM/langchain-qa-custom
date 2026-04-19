import streamlit as st
import os
from dotenv import load_dotenv
from query import get_rag_service
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="🤖 Q&A App", layout="centered")
st.title("🤖 Q&A App")

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_service" not in st.session_state:
        # get_rag_service is cached by @st.cache_resource in query.py
        with st.spinner("Initializing RAG Service..."):
            st.session_state.rag_service = get_rag_service()

def display_chat():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

def handle_user_input():
    """Handle user input and generate response."""
    if prompt := st.chat_input("Ask a question about the documents:"):
        # 1. Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_service.query(
                    prompt, 
                    st.session_state.messages
                )
                answer = response["answer"]
                st.markdown(answer)
        
        # 3. Append to history
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.messages.append(AIMessage(content=answer))

if __name__ == "__main__":
    init_session_state()
    display_chat()
    handle_user_input()
