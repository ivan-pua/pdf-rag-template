from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import KNNRetriever

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

# 4. Save embeddings in vector store
vector_store_folder_path = "faiss_index"
if  os.path.isdir(vector_store_folder_path):

    print("\n4. Loading existing vector store... ")
    vectordb = FAISS.load_local(
        vector_store_folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

else:
    print("\n4. Create new vector store... ")
    vectordb = FAISS.from_documents(split_documents, embeddings)
    vectordb.save_local(vector_store_folder_path)

# 5. Retriever
print("\n5. Creating a retriever... ")
# by default uses cosine similarity
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

query = "How are companies using LLMs?"
print(f"Question: {query}\n")

# retrieve the most relevant documents
result = retriever.invoke(query)

print("Relevant Documents: ")
for r in result:
    retrieved_doc = r.page_content  # return the top1 retrieved result
    print(retrieved_doc, "\n")
