
import os
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

from colorama import Fore
import warnings

from retrieval.fusion import reciprocal_rank_fusion

warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatOpenAI(model="AliBaba/Qwen3.6-27B")

# RAG-Fusion
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)


# GENERATION
prompt_template = """Answer the following question based on this context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

#### INDEXING ####

# Load blog
#loader = WebBaseLoader(
#    web_paths=("https://en.wikipedia.org/wiki/Retrieval-augmented_generation",),
#    requests_kwargs={
#        "headers": {
#            "User-Agent": "Mozilla/5.0",
#        },
#        "timeout": 20,
#    },
#    bs_kwargs={
#        "parse_only": bs4.SoupStrainer(id="mw-content-text"),
#    },
#)
#blog_docs = loader.load()
loader = DirectoryLoader(
    "/home/hojat/Documents/ww/LangChain-OpenTutorial/docs/12-RAG",
    glob="03-RAG-Advanced.md",
    loader_cls=TextLoader,
)
blog_docs = loader.load()

markdown_text = blog_docs[0].page_content

# Make splits
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

header_docs = header_splitter.split_text(markdown_text)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

splits = text_splitter.split_documents(header_docs)
#for i, doc in enumerate(splits):
#    print("=" * 60)
#    print(f"Chunk {i}")
#    print(doc.page_content)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

# Index
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####
generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

# Query
def query(query):
    question = "What are the two main components of a typical RAG application?"
    
    # RAG Chain
    retrieval_chain_rag_fusion = (
            generate_queries
            | retriever.map()
            | reciprocal_rank_fusion
    )

    #docs = retrieval_chain_rag_fusion.invoke({"question": question})
    #print(f"{Fore.CYAN}{len(docs)} documents retrieved {Fore.RESET}")
    
    # GENERATION Chain
    final_rag_chain = (
            {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
            )

    return final_rag_chain.invoke({"question": question})
