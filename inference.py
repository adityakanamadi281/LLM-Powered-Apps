from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

from agno.agent import Agent
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.ollama import Ollama

URL = ["https://api.freshservice.com/#ticket_attributes"]

loader = WebBaseLoader(URL)
data=loader.load()
print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

chunks=text_splitter.split_documents(data)
print(len(chunks))

embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")

client=QdrantClient(path="/tmp/app")
collection_name = "agent-rag"

try:
  collection_info = client.get_collection(collection_name=collection_name)
except (UnexpectedResponse, ValueError):
  client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=1024,distance=Distance.COSINE),
  )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vector_store.add_documents(documents=chunks)

retriever = vector_store.as_retriever()
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

agent =Agent(
    model=Ollama(id="llama3.2"),
    #knowledge_base=knowledge_base,
    description="Answer to the questions from the knowledge base",
    markdown=True,
    #search_knowledge_base=True,
)

user_query = "Give me the curl command to create a ticket"


agent.print_response(user_query, stream=True)
#response=agent.run(user_query).content
#print(response)

