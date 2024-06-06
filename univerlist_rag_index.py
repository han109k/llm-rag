import json
import os
from pathlib import Path

import html2text
import qdrant_client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up the OpenAI model
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

h_parser = html2text.HTML2Text()
# read from file called univerlist.json in main directory
file_path = "./univerlist.json"
data = json.loads(Path(file_path).read_text())

if os.path.exists("univerlist_rag.txt"):
    os.remove("univerlist_rag.txt")

uni_set = set()
for university in data["query"]:
    country = university["country"]
    province = university["province"]
    uni_name = university["uname"]
    faculty = university["faculty"]
    pname = university["pname"]
    lang = university["language"]
    p17 = university["mark17"]
    p18 = university["mark18"]
    p19 = university["mark19"]
    p20 = university["mark20"]
    p21 = university["mark21"]
    p22 = university["mark"]
    s17 = university["order17"]
    s18 = university["order18"]
    s19 = university["order19"]
    s20 = university["order20"]
    s21 = university["order21"]
    s22 = university["order"]
    sch = university["scholar"]

    if sch in ["%25", "%50", "%75", "%100"]:
        sch = sch + " Burslu"

    text_to_embed = f"Üniversite: {uni_name} {faculty}, Bölüm: {pname} ({lang}), {sch}, 2022 Puanı: {p22}, 2022 Sıralaması: {s22}, 2021 Puanı: {p21}, 2021 Sıralaması: {s21}, 2020 Puanı: {p20}, 2020 Sıralaması: {s20}, 2019 Puanı: {p19}, 2019 Sıralaması: {s19}, 2018 Puanı: {p18}, 2018 Sıralaması: {s18}, 2017 Puanı: {p17}, 2017 Sıralaması: {s17}"

    content = university["content"]
    if content is not None and uni_name not in uni_set:
        content = content.replace("\n", " ")
        content = content.replace("\r", " ")
        content = h_parser.handle(content)
        text_to_embed += content
        uni_set.add(uni_name)
    else:
        text_to_embed = f"Konum: {country}, {province}, " + text_to_embed + "\n"

    # write to file called univerlist_rag.txt in main directory
    with open("univerlist_rag.txt", "a") as f:
        f.write(text_to_embed)


docs = TextLoader("univerlist_rag.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant = Qdrant.from_documents(
    documents,
    embedding=embeddings,
    url="http://localhost:6333",
    prefer_grpc=True,
    collection_name="univerlist_rag",
    force_recreate=True,  # * Set to True to recreate the collection
)

client = qdrant_client.QdrantClient("http://localhost:6333")
doc_store = Qdrant(
    client=client, embeddings=embeddings, collection_name="univerlist_rag"
)

found_docs = doc_store.similarity_search(
    query="ODTÜ Bilgisayar Mühendisliği 2022 Puanı nedir?", k=2
)
print(found_docs)
