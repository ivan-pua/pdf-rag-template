from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Loaded PDF
file_path = "files/Databricks-Big-Book-Of-GenAI.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

print(len(documents))
print(documents[10].page_content[0:100])
print(documents[10].metadata)

# 2. Split document into chunks - for precise matching and fitting context window
# text_splitter = Recus

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents=documents)

print("\nAfter Splitting: \n")
print(len(split_documents))
print(split_documents[10])

