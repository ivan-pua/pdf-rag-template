from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import KNNRetriever
from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering

load_dotenv()  # takes variables from .env file


def generate_response(query, uploaded_files):
    # 1. Loaded PDF
    print("1. Loading PDF files...")
    # file_path = "files/databricks-llm.pdf"
    loader = PyPDFLoader(f"files/{uploaded_files.name}")

    documents = loader.load()

    print(len(documents))
    print(documents[0].page_content[0:100])
    print(documents[0].metadata)

    # 2. Split document into chunks - for precise matching and fitting context window
    print("\n2. Chunking documents... ")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents=documents)
    print(len(split_documents))
    print(split_documents[0])

    voyage_key = os.getenv("VOYAGE_API_KEY")

    # 3. Create embeddings
    print("\n3. Create embeddings...")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=voyage_key, model="voyage-2",
    )

    # 4. Save embeddings in vector store
    vector_store_folder_path = "faiss_index"
    if os.path.isdir(vector_store_folder_path):

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

    # query = "How are companies using LLMs?"
    print(f"Question: {query}\n")

    # retrieve the most relevant documents
    docs = retriever.invoke(query)

    context = '\n\n'.join([doc.page_content for doc in docs])
    print(context)

    # 6. Set up LLM
    # Specify the model name you want to use
    model_name = "deepset/tinyroberta-squad2"

    # # Download a tokenizer object by loading the pretrained "Intel/dynamic_tinybert" tokenizer.
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    # # Download a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a question-answering pipeline using the model and tokenizer
    question_answerer = pipeline(
        task="question-answering",
        model=model,
        tokenizer=tokenizer,
        temperature=0.5,
        max_new_tokens=256,
        repetition_penalty=1.5,
        do_sample=True,
        device=0
    )

    result = question_answerer(question=query, context=context)
    print(
        f"\nAnswer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

    return result['answer']
