import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the OpenAI model
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

docs = TextLoader("transcription.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
documents = text_splitter.split_documents(docs)

# Set up the OpenAI Embeddings
embeddings = OpenAIEmbeddings()  # Default embedding model is "text-embedding-ada-002"

qdrant = Qdrant.from_documents(
    documents,
    embedding=embeddings,
    url="http://localhost:6333",
    prefer_grpc=True,
    collection_name="my_documents",
    force_recreate=True,  # Set to True to recreate the collection
)

query = "What is synthetic intelligence?"
# found_docs = qdrant.similarity_search(
#     query
# )  # Search for similar documents using cosine similarity
# print(found_docs)

retreiver = qdrant.as_retriever()

setup = RunnableParallel(context=retreiver, question=RunnablePassthrough())

# Prompt Template - Simple way to define and reuse prompts
template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
parser = StrOutputParser()  # Needed to parse the 'content' key from the response

chain = setup | prompt | model | parser
response = chain.invoke(query)
print("Response: ", response)

response = chain.invoke("What is this podcast about?")
print("Response: ", response)
