import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import Chroma
from utils.process_documents import process_pdf_documents
from colorama import Fore
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

template: str = """/
    You are customer support specialist /
    question: {question}. You assist users with general inquiries based on {context} /
    and technical isssue. /
    """
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# model = ChatOpenAI()
model = ChatOpenAI(model="OpenAI/GPT-OSS-120b")


def load_documents():
    """load a file form path, split it into chunks, embed each chunk and load it into the vector store."""
    loader = TextLoader("./docs/user-manual.txt")
    raw_text = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    return text_splitter.split_documents(raw_text)


def load_embeddings(documents):
    """Create a vector store from a set of documents."""
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma.from_documents(documents, embeddings)
    return db.as_retriever()


# retriever = load_embeddings()


def generate_response(retriever, query):
    """Generate a response to a user query."""

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )
    return chain.invoke(query)


def query(query):
    """Query the model and return the response."""
    documents = load_documents()
    retriever = load_embeddings(documents)
    response = generate_response(retriever, query)
    return response


def start():
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
    )
    print(Fore.BLUE + "\n\x1b[3m" + instructions + "\x1b[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        return start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:
            response = query(user_input)
            print(f"{Fore.GREEN}==== ANSWER ====")
            print(response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
