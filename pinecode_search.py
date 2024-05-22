from flask import Flask, render_template, request, redirect, url_for
import os
import re
import string
import contractions
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'home/nitinrathod/Music/flask/LLM/Advanced_AI/Atomic habits.pdf'

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment='us-west1-gcp')  # Adjust the environment if necessary
index_name = "pdf-data"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric='euclidean')

index = pinecone.Index(index_name)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def embed_text(text):
    openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embeddings = openai_embeddings.embed_query(text)
    return embeddings

def retrieve_top_matching_content(query_text, top_k=10):
    text = clean_text(query_text)
    embeddings = embed_text(text)
    if embeddings:
        result = index.query(embeddings, top_k=top_k, namespace="pdf_text", include_metadata=True)
        return result
    else:
        print("Embedding failed.")
        return None

def answer_generator(raw_input, pdf_text):
    content = f"you are AI ASSISTANT for Human habits, your work is suggestions for human good and bad habits on the basis of the atomic habits book, analyze the user query very carefully and respond accordingly. Your response should be in proper formatting and in bullet points if needed. Do not use numbers. If there is any heading, then it will be in bold <b> tag. Give a response only related to the above asked query from the given text content only. Here is the content - '{pdf_text}', and the user's query is - '{raw_input}', explain it in short."
    pdf_chat_completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    pdf_response = pdf_chat_completion.choices[0].message.content
    formatted_response = format_response(pdf_response)
    print("Data fetched successfully from the pdf\n")
    return formatted_response

def format_response(answer_text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', answer_text)
    if urls:
        for url in urls:
            anchor_tag = f'<a href="{url}" target="_blank">{url}</a>'
            answer_text = answer_text.replace(url, anchor_tag)
    return answer_text

@app.route("/Predict", methods=["GET, POST"])
def index():
    if request.method == "POST":
        query_text = request.form["query"]
        file = request.files.get("file")
        url = request.form.get("url")

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the uploaded file (PDF, image, etc.)
            # For example, you can extract text from the file here
            extracted_text = extract_text_from_file(file_path)
            response = main(query_text, extracted_text)
        elif url:
            # Process the URL (fetch content from the URL)
            url_content = fetch_content_from_url(url)
            response = main(query_text, url_content)
        else:
            response = main(query_text)

        return render_template("index.html", query=query_text, response=response)

    return render_template("index.html", query=None, response=None)

def main(query_text, additional_text=""):
    top_matches = retrieve_top_matching_content(query_text)
    page_text = additional_text
    if top_matches:
        for match in top_matches["matches"]:
            page_text += match["metadata"]["page_text"]
    else:
        print("No matching content found.")
    response = answer_generator(query_text, page_text)
    return response

def extract_text_from_file(file_path):
    # Implement your text extraction logic here (e.g., using PyPDF2, pytesseract for OCR, etc.)
    return "Extracted text from the file"

def fetch_content_from_url(url):
    # Implement your logic to fetch and process content from the URL here
    return "Fetched content from the URL"

if __name__ == "__main__":
    app.run(debug=True)
