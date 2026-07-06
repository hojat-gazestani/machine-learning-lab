
import os
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langchain_core.pydantic_v1 import BaseModel, Field

from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

'''
THE STEPS
1. Let’s skip the knowledge refinement phase as a first pass. This can be added back as a node, if desired.
2. If any documents are irrelevant, let’s opt to supplement retrieval with web search.
3. We’ll use Tavily Search for web search.
4. Let’s use query re-writing to optimize the query for web search.
'''

load_dotenv()

llm = ChatOpenAI(model="Google/Gemma-4-31B-it")

#### INDEXING ####
urls = [
    "https://raw.githubusercontent.com/hojat-gazestani/Notes/refs/heads/main/haproxy/01-Conepts/02-Tune%20Timeouts.md",
    "https://raw.githubusercontent.com/hojat-gazestani/Notes/refs/heads/main/haproxy/01-Conepts/03-Proxy.md",
    "https://raw.githubusercontent.com/hojat-gazestani/Notes/refs/heads/main/haproxy/01-Conepts/04-Load%20balancer.md"
]

loaded_docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in loaded_docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

#### Retrieval Grader : Retrieval Evaluator ####
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
 
    def get_score(self) -> str:
        """Return the binary score as a string."""
        return self.binary_score

#def get_score(self) -> str:
#    """Return the binary score as a string."""
#    return self.binary_score
def get_score(self) -> str:
    return self.binary_score

# LLM with function call 
structured_llm_grader = llm.with_structured_output(
        GradeDocuments, 
        method="function_calling",
)

# Prompt 
system_template = """
You are a retrieval evaluator.

Determine whether the retrieved {documents} contains enough information to answer the user's {question}.

Reply only with:

- yes
- no

Mark "yes" if the document is directly or semantically relevant.
Mark "no" if it is unrelated or insufficient.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["documents", "question"],
    template="{question}",
)
grader_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

### Question Re-writer - Knowledge Refinement ####
# Prompt 
prompt_template = """Given a user input {question}, Your task is re-write or rephrase the question to optimize the query in order to improve the content generation"""

system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question"],
    template="{question}",
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

### Web Search Tool - Knowledge Searching ####
web_search_tool = TavilySearchResults(k=3) 

#### Generate Answer  ####
# Prompt
#prompt = hub.pull("rlm/rag-prompt")
prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. if you don't konw the answer, just say that you don't know. Use three sentences maximum and the answer concis.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# Retrieve and assess
def assess_retrieved_docs(query):
    """Retrieve and assess the relevance of documents to a given query."""
    retrieval_grader = grader_prompt | structured_llm_grader | get_score
    retrieved_docs = retriever.get_relevant_documents(query)
    #doc_txt = docs[0].page_content
    doc_txt = "\n\n".join(doc.page_content for doc in retrieved_docs)
    binary_score = retrieval_grader.invoke({"question": query, "documents": doc_txt})
    return binary_score, retrieved_docs

# Rewrite and optimize 
def rewrite_query(query):
    """Rewrite and optimize a given user query for the model."""
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": query})

# Search the web
def search_web(query):
    """Search the web for complimentary information."""
    web_results = web_search_tool.invoke({"query": query})
    #web_docs = "\n".join([doc["content"] for doc in docs])
    web_docs = "\n".join(
        doc["content"] if isinstance(doc, dict) else str(doc)
        for doc in web_results
    )
    return Document(page_content=web_docs)

def generate_answer(docs, query):      
    # Chain
    rag_chain = prompt | llm |StrOutputParser()

    # Run
    if isinstance(docs, list):
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = docs.page_content

    print("=" * 80)
    print(context[:1000])
    print("=" * 80)
    return rag_chain.invoke({"context": docs, "question": query})



def query(query):
    """Query the model with a question and assess the relevance of retrieved documents."""
    #query = "RAG"
    #query = "What HAProxy load balancing algorithm?"
    binary_score, docs = assess_retrieved_docs(query)

    print(f"Relevance score: {binary_score}")
    if binary_score == "yes":
        return generate_answer(docs, query)

    print(f"{Fore.YELLOW}Rewriting the query for content generation.{Fore.RESET}")
    optimized_query = rewrite_query(query)
    print(f"{Fore.MAGENTA}Retrieved documents are irrelevant. Searching the web for additional information.{Fore.RESET}")
    docs = search_web(optimized_query)
    return generate_answer(docs, query)
