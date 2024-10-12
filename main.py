from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import KNNRetriever
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from time import time
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv() # takes variables from .env file

# 1. Loaded PDF
print("1. Loading PDF files...")
file_path = "files/databricks-llm.pdf"
loader = PyPDFLoader(file_path)

documents = loader.load()

print(len(documents))
print(documents[0].page_content[0:100])
print(documents[0].metadata)

# 2. Split document into chunks - for precise matching and fitting context window
print("\n2. Chunking documents... ")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(documents=documents)
print(len(split_documents))
print(split_documents[0])

voyage_key = os.getenv("VOYAGE_API_KEY")

# 3. Create embeddings
print("\n3. Create embeddings...")
embeddings = VoyageAIEmbeddings(
    voyage_api_key=voyage_key, model="voyage-2",
)

# sample_docs = [
#     "John works in TPG Telecom."
# ]

# documents_embds = embeddings.embed_documents(sample_docs)

# print(documents_embds[0][:5])

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

# 6. Set up LLM
# Specify the model name you want to use
model_name = "facebook/opt-125m"

# # Download a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Download a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer,
    temperature=0.1,
    max_new_tokens=256,  
    repetition_penalty=1.9,
    do_sample=True,
    device=0
)
# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
)

prompt_template = """
<|system|>
Answer the question based on the context below. Use the following context to help:

{context}

Question:
{question}

Answer:

"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

start_time = time()

print(rag_chain.invoke(query))
end_time = time()

print(f"Time taken: {end_time -start_time:.2f} seconds.")
