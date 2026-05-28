import bs4
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import Chroma

##from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils import format_qa_pair, format_qa_pairs

from colorama import Fore
import warnings

import os

os.environ["USER_AGENT"] = "rag-learning-project"

warnings.filterwarnings("ignore")

load_dotenv()


# model = ChatOpenAI()
# llm = ChatOpenAI()
llm = ChatOpenAI(model="OpenAI/GPT-OSS-120b")

# Load wikipedia-like content
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

docs = loader.load()

print("docs:", len(docs))

## Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
print("splits:", len(splits))

## Index and load embeddings
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
)
print("Vector store created!")

## Create the vector store
retriever = vectorstore.as_retriever()
print(f"retriver: {retriever}")


# 1. DECOMPOSITION
template = """You are a helpful assistant trained to generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)


def generate_sub_questions(query):
    """generate sub questions based on user query"""
    # Chain
    generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    # Run
    sub_questions = generate_queries_decomposition.invoke({"question": query})
    questions_str = "\n".join(sub_questions)
    print(Fore.GREEN + "====== Sub-questiens ======" + Fore.RESET)
    print(Fore.WHITE + questions_str + Fore.RESET + "\n")
    return sub_questions


# 2. ANSWER SUBQUESTIONS RECURSIVELY
template = """Here is the question you need to answer:

\n --- \n {sub_question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question:

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {sub_question}
"""
prompt_qa = ChatPromptTemplate.from_template(template)


def generate_qa_pairs(sub_questions):
    """ask the LLM to generate a pair of question and answer based on the original user query"""
    q_a_pairs = ""

    for sub_question in sub_questions:
        generate_qa = (
            {
                "context": itemgetter("sub_question") | retriever,
                "sub_question": itemgetter("sub_question"),
                "q_a_pairs": itemgetter("q_a_pairs"),
            }
            | prompt_qa
            | llm
            | StrOutputParser()
        )
        answer = generate_qa.invoke(
            {"sub_question": sub_question, "q_a_pairs": q_a_pairs}
        )
        q_a_pair = format_qa_pair(sub_question, answer)
        q_a_pairs = q_a_pairs + "\n --- \n" + q_a_pair


# 3. ANSWER INDIVIDUALY

# RAG prompt = https://smith.langchain.com/hub/rlm/rag-prompt
# prompt_rag = pull("rlm/rag-prompt")

prompt_rag = ChatPromptTemplate.from_template("""
Answer the question based only on the context.

Context:
{context}

Question:
{question}
""")


def retrieve_and_rag(prompt_rag, sub_questions):
    """RAG on each sub-question"""
    rag_results = []
    for sub_question in sub_questions:
        retrieved_docs = retriever.invoke(sub_question)

        answer_chain = prompt_rag | llm | StrOutputParser()
        answer = answer_chain.invoke(
            {"question": sub_question, "context": retrieved_docs}
        )
        rag_results.append(answer)

    return rag_results, sub_questions


# SUMMARIZE AND ANSWER

# Prompt
template = """Here is a set of Q+A pairs:

{context}

to use these to synthesize an answer to the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


# Query
def query(query):
    # generate optimized answer for a given query using the improved subqueries
    # question = "What are the main components of an LLM-powered autonomous agent system?"
    queries = [
        "How is context improving AI systems",
        "What are the two main components involved in Basic Reflection",
        "Explain the steps involved in the Reflexion loop",
    ]
    sub_questions = generate_sub_questions(query)
    generate_qa_pairs(sub_questions)
    answers, questions = retrieve_and_rag(prompt_rag, sub_questions)
    context = format_qa_pairs(questions, answers)

    final_rag_chain = prompt | llm | StrOutputParser()

    return final_rag_chain.invoke({"question": questions, "context": context})
