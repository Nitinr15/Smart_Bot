from PyPDF2 import PdfReader
import re
import string
import nltk
import contractions
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    return text

def extract_text_from_pdf(pdf_file_path):
    """
    Extracts text from each page of a PDF document.

    Args:
        pdf_file_path (str): The path to the PDF file.

    Returns:
        list: A list containing text extracted from each page of the PDF document.
    """
    extracted_text = []

    # Load the PDF document
    with open(pdf_file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        
        # Check the number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        
        # Iterate through each page and read the data
        for page_num in range(num_pages):
            # Extract text from the current page
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            extracted_text.append(page_text)
    
    return extracted_text

def embed_text(text):
    try:
        # Initialize OpenAI embeddings from Langchain
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Perform embedding
        embeddings = openai_embeddings.embed_query(text)
        return embeddings
    except Exception as e:
        print("An error occurred while embedding text:", e)
        breakpoint()

        # return None

# Usage example:
pdf_file_path = "/home/nitinrathod/Music/flask/LLM/Advanced_AI/Atomic habits.pdf"
extracted_text = extract_text_from_pdf(pdf_file_path)
for page_num, page_text in enumerate(extracted_text, start=1):
    cleaned_text = clean_text(page_text)
    # embeddings = embed_text(cleaned_text)
    if embeddings:
        print("Page", page_num, "Embeddings:", embeddings)
    else:
        print("Page", page_num, "Embedding failed.")
