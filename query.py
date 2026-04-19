__import__('pysqlite3')
import sys
import os
import streamlit as st
from dotenv import load_dotenv

# SQLite shim for ChromaDB compatibility
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class RAGService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or st.secrets.get("openai_api_key")
        if not self.api_key:
            raise ValueError("OpenAI API Key not found.")
        
        self.llm = ChatOpenAI(api_key=self.api_key, model="gpt-3.5-turbo")
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self._setup_rag_chain()

    def _setup_rag_chain(self):
        # 1. Load and Split Documents
        loader = TextLoader("./docs/faq.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 2. Create Vector Store (In-memory for simplicity, can be persisted)
        self.vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

        # 3. Contextualize Question Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 4. QA Prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # 5. Build Chains
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        self.question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever, self.question_answer_chain
        )

    def query(self, input_text, chat_history):
        """
        Executes a query against the RAG chain.
        chat_history should be a list of BaseMessage objects.
        """
        response = self.rag_chain.invoke({
            "input": input_text,
            "chat_history": chat_history
        })
        return response

# For backward compatibility and specialized Streamlit caching
@st.cache_resource
def get_rag_service():
    return RAGService()

def query(query_text):
    """
    Legacy wrapper for existing app.py/main.py logic.
    Note: This uses a global history which is discouraged for multi-user apps.
    """
    if 'global_chat_history' not in globals():
        global global_chat_history
        global_chat_history = []
    
    service = get_rag_service()
    # Convert string history to LangChain messages if needed, 
    # but here we'll just handle the invocation.
    response = service.query(query_text, global_chat_history)
    
    # Update global history (Legacy behavior)
    global_chat_history.extend([
        HumanMessage(content=query_text),
        AIMessage(content=response["answer"])
    ])
    return response
