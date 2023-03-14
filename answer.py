# Imports
from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import pickle
import os


def getAnswer(apikey, prompt, temperature):
    """
    prompt: The prompt to be answered
    temperature: 
    """
    
    # Initialize model
    llm = OpenAI(model_name='text-davinci-003', temperature=temperature, openai_api_key=apikey)
    
    # Get cached OpenAI embeddings
    embeddings = get_cached_openai_embeddings()
    
    # Get stored Chroma instance
    docsearch = load_existing_chromadb(embeddings)
    
    # Use LangChain's VectorDBQA chain for question answering against our vector database
    qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch) # picks 4 documents by default
    try:
        result = qa_chain.run(prompt)
    except:
        # TODO: check for wrong api key here somehow
        result = "Something went wrong"
        
    return result
    
def load_existing_chromadb(embeddings):
    """Checks if there exists a chromadb instance locally and fetches embeddings for our document from it.
    Args:
        embeddings: The output of the embedding_function i.e OpenAI's embeddings
        
    Returns:
        docsearch: The embeddings in the form of a database

    """
    persist_directory = 'db'
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return docsearch

def get_cached_openai_embeddings():
    """"""
    # Check if already exists
    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
            return embeddings
        
    # else:
        # TODO: Create embeddings using OpenAI's model