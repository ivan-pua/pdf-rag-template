from langchain_community.document_loaders import PyPDFLoader

file_path = "files/Databricks-Big-Book-Of-GenAI.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

print(len(documents))
print(documents[1])