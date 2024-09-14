from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import aiohttp
import os
import ssl

load_dotenv() # takes variables from .env file

# 1. Loaded PDF
print("1. Loading PDF files...")
file_path = "files/compact-guide-to-large-language-models.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

print(len(documents))
print(documents[0].page_content[0:100])
print(documents[0].metadata)

# 2. Split document into chunks - for precise matching and fitting context window
# text_splitter = Recus

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents=documents)

print("\n2. Chunking documents... ")
print(len(split_documents))
print(split_documents[0])

key = os.getenv("VOYAGE_API_KEY")

# 3. Create embeddings
print("\n3. Create embeddings...")
embeddings = VoyageAIEmbeddings(
    voyage_api_key=key, model="voyage-2",
)

sample_docs = [
    "John works in TPG Telecom."
]

documents_embds = embeddings.embed_documents(sample_docs)

print(documents_embds[0][:5])

