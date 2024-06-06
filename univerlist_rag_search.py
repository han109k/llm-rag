import os

import numpy as np
import qdrant_client
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from trulens_eval import Feedback, Tru, TruChain
from trulens_eval.app import App
from trulens_eval.feedback.provider import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tru = Tru()
tru.reset_database()

# Set up the OpenAI model
model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

embeddings = OpenAIEmbeddings()

client = qdrant_client.QdrantClient("http://localhost:6333")
doc_store = Qdrant(
    client=client, embeddings=embeddings, collection_name="univerlist_rag"
)
found_docs = doc_store.similarity_search(
    "yaşar üni hakkında bilgi verebilir misin?", k=4
)
print(found_docs)

# prompt = hub.pull("rlm/rag-prompt")

template = """
Aşağıdaki soruya 'Context' baz alarak cevap ver.
Cevap verirken Türkçe dilini kullan. Cevaplamadan önce üzerinde düşün.
Cevaplarken sonucu detaylandırabilirsin.

Context: {context}

Soru: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retreiver = doc_store.as_retriever()

setup = RunnableParallel(context=retreiver, question=RunnablePassthrough())
parser = StrOutputParser()

chain = setup | prompt | model | parser

# response = chain.invoke("yaşar üniversitesi eğitim ücretleri ne?")
# print("\033[36m Response: ", response)

####!!!! Evaluation of the model with TruLens !!!!####
# https://medium.com/@glenpatzlaff/raw-json-to-measurable-rag-insights-in-a-matter-of-minutes-with-langchain-and-trulens-f36e4415b079
# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
context = App.select_context(chain)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons)
    .on(context.collect())  # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Feedback(provider.relevance).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons)
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(
    chain,
    app_id="Univerlist_RAG",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

prompts = [
    "Who is the Vice President?",
    "How do I reach Senator Duckworth?",
    "yaşar üni hakkında bilgi verebilir misin?",
    "Yaşar Üniversitesi ücretlendirme nasıldır?",
]

with tru_recorder:
    for prompt in prompts:
        result = chain.invoke(prompt)

records, feedback = tru.get_records_and_feedback(app_ids=["Chain1_ChatApplication"])

records.head()

tru.get_leaderboard(app_ids=["Chain1_ChatApplication"])

tru.run_dashboard()  # open a local streamlit app to explore
