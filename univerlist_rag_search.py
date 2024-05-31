import json
import os
from pathlib import Path

import html2text
import numpy as np
import qdrant_client
from dotenv import load_dotenv
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from trulens_eval import Feedback, Tru, TruChain
from trulens_eval.feedback.provider import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the OpenAI model
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

embeddings = OpenAIEmbeddings()

client = qdrant_client.QdrantClient("http://localhost:6333")
doc_store = Qdrant(
    client=client, embeddings=embeddings, collection_name="univerlist_rag"
)
found_docs = doc_store.similarity_search(
    "Yaşar Bilgisayar Mühendisliği 2022 Puanı nedir?", k=2
)
print(found_docs)

prompt = hub.pull("rlm/rag-prompt")
retreiver = doc_store.as_retriever()

setup = RunnableParallel(context=retreiver, question=RunnablePassthrough())
parser = StrOutputParser()

chain = setup | prompt | model | parser

response = chain.invoke("Yaşar Bilgisayar Mühendisliği 2022 Puanı nedir?")
print("Response: ", response)
