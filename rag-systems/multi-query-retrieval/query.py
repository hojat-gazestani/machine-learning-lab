
import os
import bs4
from dotenv import load_dotenv
#from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Multi-query RAG

## Step 1. Create the LLM
llm = ChatOpenAI(model="AliBaba/Qwen3.6-27B")


## Step 2. Create the prompt
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
multi_query_prompt = ChatPromptTemplate.from_template(template)


rag_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# INDEXING,

## Step 3. Download the document
loader = WebBaseLoader(
    web_paths=("https://en.wikipedia.org/wiki/Retrieval-augmented_generation",),
    requests_kwargs={
        "headers": {
            "User-Agent": "Mozilla/5.0",
        },
        "timeout": 20,
    },
)
blog_docs = loader.load()

## Step 4. Split the document
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

## Index

## Step 5. Embed every chunk
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


## Step 6. Build the vector database
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

## Step 7. Create a retriever
retriever = vectorstore.as_retriever()


## Step 8. Build the Multi-query chain
def generate_multi_queries():
    """ generate multiple queries given a user input """
    return (
        multi_query_prompt
        | llm
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

# GENERATION 
def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and covert each Document to string
    flattened_docs = [ dumps(doc) for sublist in documents for doc in sublist]

    # Get unique documents
    unique_docs = list(set(flattened_docs))

    return [loads(doc) for doc in unique_docs]

## Step 9. User asks a question
def query(query):
    question = "how can we make retrieval robust to variability in user input?"

    # Retrieval Multi-query Chain
    ## Step 11. Retriever
    retrieval_chain = generate_multi_queries() | retriever.map() | get_unique_union
    docs = retrieval_chain.invoke({"question": question})
    print(f"{Fore.CYAN}{len(docs)} documents retrieved {Fore.RESET}")


    ## Step 10. Parse
    ## Step 13. Generation 
    final_rag_chain = (
            {"context": retrieval_chain, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    return final_rag_chain.invoke(query)

