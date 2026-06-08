from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi.middleware.cors import CORSMiddleware
import chromadb

load_dotenv()

app = FastAPI(title="Logiqbits AI Wrapper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== LLM CONFIG ======================
llm = OpenAILike(
    model="LFM2.5-1.2B-Instruct-Q4_K_M",
    api_base="http://localhost:8080/v1",
    api_key="fake-key-not-needed",
    temperature=0.7,
    max_tokens=1024,
    is_chat_model=True,
    context_window=8192,
)

# ====================== EMBEDDING CONFIG ======================
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# ====================== COLLECTION CONFIG ======================

ACTIVE_COLLECTION = "crm"  # options: "crm", "hrm", "core", "direct"

SYSTEM_PROMPTS = {
    "crm": """You are a CRM specialist AI assistant for Logiqbits SaleSense CRM.
You help users with customer relationship management, sales pipelines, lead tracking, and deal management.
Be concise, accurate, and solution-oriented. Use the provided context when available.""",

    "hrm": """You are an HRM specialist AI assistant for Logiqbits HCM.
You help users with human resource management, payroll, attendance, employee records, and performance tracking.
Be concise, accurate, and solution-oriented. Use the provided context when available.""",

    "core": """You are a core business AI assistant for Logiqbits.
You help users with FiNext Lite ERP, finance management, inventory, and general business operations.
Be concise, accurate, and solution-oriented. Use the provided context when available.""",

    "direct": """You are a helpful, professional AI assistant for Logiqbits (https://logiqbits.com/).
You specialize in all Logiqbits business solutions including FiNext Lite, ShopAP, SaleSense CRM, HCM, Automation, and Cloud services.
Be concise, accurate, and solution-oriented. Use the provided context when available.""",
}

# ====================== LOAD ACTIVE RETRIEVER ======================
def load_retriever(collection_name: str):
    chroma_client = chromadb.PersistentClient(path=f"./chroma_db/{collection_name}")
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index.as_retriever(similarity_top_k=3)

retriever = load_retriever(ACTIVE_COLLECTION)
system_prompt = SYSTEM_PROMPTS[ACTIVE_COLLECTION]

print(f"Active collection: {ACTIVE_COLLECTION}")

# ====================== MODELS ======================
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    response: str
    status: str = "success"
    sources: Optional[List[str]] = None
    collection: str = ACTIVE_COLLECTION

# ====================== ROUTE ======================
@app.post("/ai", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        retrieved_nodes = retriever.retrieve(request.message)
        context_text = "\n\n".join([n.get_content() for n in retrieved_nodes])
        sources = list(set([
            n.metadata.get("file_name", "unknown")
            for n in retrieved_nodes
            if n.metadata
        ]))

        messages: List[ChatMessage] = []

        system_with_context = system_prompt
        if context_text.strip():
            system_with_context += f"\n\n--- Relevant Context ---\n{context_text}\n------------------------"

        messages.append(ChatMessage(role="system", content=system_with_context))

        if request.history:
            for msg in request.history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(ChatMessage(role=role, content=content))

        messages.append(ChatMessage(role="user", content=request.message))

        response = llm.chat(messages)

        return ChatResponse(
            response=response.message.content,
            status="success",
            sources=sources if sources else None,
            collection=ACTIVE_COLLECTION,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "active_collection": ACTIVE_COLLECTION,
        "message": f"Logiqbits AI Wrapper — Collection: {ACTIVE_COLLECTION}"
    }