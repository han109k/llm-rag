import os

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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
embeddings = OpenAIEmbeddings()

db =  Qdrant.from_documents(documents, embeddings, "http://localhost:6333")
query = "What did the president say about Ketanji Brown Jackson"
res = db.similar_documents(query)
print(docs[0].page_content)