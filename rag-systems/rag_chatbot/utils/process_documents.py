import os
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from unstructured_client.models import shared
from unstructured.documents.elements import CompositeElement, Table
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from pdfminer.high_level import extract_text
from langchain_openai import ChatOpenAI
from unstructured_client import UnstructuredClient
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from colorama import Fore
from PIL import Image
import base64
import io

load_dotenv()

filepaths = ["./docs/text-splitters.pdf"]
pdf_elements = []
documents = []
UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY")
model = ChatOpenAI()
gpt4 = ChatOpenAI(model="gpt-4-turbo", max_tokens=1024)

# init client


def element_to_document(element):
    if isinstance(element, CompositeElement):
        content = element.text  # Assuming CompositeElement has a text attribute
    elif isinstance(element, Table):
        content = str(
            element.to_dict()
        )  # Assuming Table has a to_list method to convert to a list
    else:
        content = str(element)  # Fallback to string representation for unknown types
    return Document(page_content=content)


# Extracts the elements from the PDF
def extract_elements_from_pdf(filepaths):
    """extract elements from pdf with partition_pdf"""
    for filepath in filepaths:
        pdf_elements = partition_pdf(
            filename=filepath,
            strategy="hi_res",
            model="yolox",
            infer_table_structure=True,
        )

    documents.append(pdf_elements)


def process_pdf_documents():
    """Process the PDF documents and extract elements."""
    print("Processing PDF documents...")
    extract_elements_from_pdf(filepaths)

    flattened_elements_list = [element for sublist in documents for element in sublist]
    docs = [element_to_document(element) for element in flattened_elements_list]
    _ = [print(doc.page_content) for doc in docs]
    return docs
