from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import uuid


def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    return embeddings

    
def create_vectorstore(chunks, embedding_function, vectorstore_path):
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks,ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)
    
    Chroma.from_documents(documents=unique_chunks,
                        embedding=embedding_function,
                        persist_directory= vectorstore_path)
    
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


if __name__ == '__main__':
    load_dotenv()
    print("Chatting...")


llm = ChatMistralAI(model="mistral-large-latest",api_key=os.environ["MISTRAL_API_KEY"])

loader = PyPDFLoader('data/The Impact of Artificial Intelligence on Modern Healthcare.pdf')
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               length_function=len,
                                               separators=["\n\n","\n"," "])

chunks = text_splitter.split_documents(pages)

embedding_function = get_embedding_function()

save_vector_store = create_vectorstore(chunks=chunks,
                                 embedding_function=embedding_function,
                                 vectorstore_path="vectorstore_chroma")

load_vector_store = Chroma(persist_directory="vectorstore_chroma",embedding_function=embedding_function)
retriever = load_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING

Strict condition: Only answer without any additional text

{context}

---

Answer the question based on the above context: {question}
"""


prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)


response = rag_chain.invoke("What is the article title?")

print(response)