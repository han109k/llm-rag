import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader("./").load_data()

llm = OpenAI(model="gpt-3.5-turbo")

# Build index with a chunk_size of 512
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
vector_index = VectorStoreIndex(nodes)

query_engine = vector_index.as_query_engine()

response_vector = query_engine.query("What is synthetic intelligence?")
print(response_vector)
