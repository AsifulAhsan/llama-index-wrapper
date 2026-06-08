from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb


COLLECTION_NAME = "crm"
DOCS_PATH = "docs"         
# ----------------------------------------------------

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(DOCS_PATH).load_data()
print(f"Loaded {len(documents)} document(s) for '{COLLECTION_NAME}'")

chroma_client = chromadb.PersistentClient(path=f"../../chroma_db/{COLLECTION_NAME}")
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

print(f"Done — indexed into chroma_db/{COLLECTION_NAME}")