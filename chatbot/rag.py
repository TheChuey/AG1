from pathlib import Path
from typing import Optional, List
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    from langchain_community.vectorstores import Chroma
    import chromadb  # verify chromadb availability
    CHROMA_AVAILABLE = True
except Exception:
    Chroma = None  # Fallback when Chroma or chromadb not available
    CHROMA_AVAILABLE = False
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from chatbot.config import Config


class KnowledgeManager:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vector_db: Optional[Chroma] = None
        # Fallback storage for when Chroma is not available
        self._fallback_chunks: List[Document] = []
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=self.config.ollama_base_url,
        )
        self.llm = OllamaLLM(
            model=self.config.model,
            temperature=self.config.temperature,
            base_url=self.config.ollama_base_url,
        )

    def load_pdf(self, file_path: str) -> str:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        persist = self.config.get("chroma_persist_dir")
        if persist:
            Path(persist).mkdir(parents=True, exist_ok=True)
        if not CHROMA_AVAILABLE:
            # Fallback: store chunks in memory
            self._fallback_chunks = chunks
            self.vector_db = None
            return f"Loaded {len(chunks)} chunks from {Path(file_path).name} (fallback storage)"
        # Attempt to use Chroma; fallback to in‑memory store on any error
        try:
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist,
            )
            return f"Loaded {len(chunks)} chunks from {Path(file_path).name}"
        except Exception as e:
            # Log the exception for debugging (could integrate with logger)
            print(f"Chroma init failed ({e}); using fallback storage.")
            self._fallback_chunks = chunks
            self.vector_db = None
            return f"Loaded {len(chunks)} chunks from {Path(file_path).name} (fallback storage)"
        

    def load_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [])]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        persist = self.config.get("chroma_persist_dir")
        if not CHROMA_AVAILABLE:
            # Fallback: store chunks in memory
            self._fallback_chunks = chunks
            self.vector_db = None
            return f"Loaded {len(chunks)} text chunks (fallback storage)"
        # Attempt to use Chroma; fallback to in‑memory store on any error
        try:
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist,
            )
            return f"Loaded {len(chunks)} text chunks"
        except Exception as e:
            print(f"Chroma init failed ({e}); using fallback storage.")
            self._fallback_chunks = chunks
            self.vector_db = None
            return f"Loaded {len(chunks)} text chunks (fallback storage)"
        
    def query(self, question: str, k: int = 3) -> str:
        # If Chroma is available, use it for retrieval
        if self.vector_db is not None:
            retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
        elif self._fallback_chunks:
            # Simple fallback: take first k chunks
            docs = self._fallback_chunks[:k]
            context = "\n\n".join(d.page_content for d in docs)
        else:
            return "No documents loaded. Use load_pdf() or load_texts() first."
        prompt = f"""You are an expert assistant. Use ONLY this context:

{context}

Question:
{question}

Answer:"""
        return self.llm.invoke(prompt)

    def is_ready(self) -> bool:
        """Return ``True`` if the knowledge manager has any data loaded.

        The original implementation only considered a Chroma vector store, which
        broke the fallback‑in‑memory path.  Now we also treat the in‑memory
        ``_fallback_chunks`` list as a valid source of documents.
        """
        return self.vector_db is not None or bool(self._fallback_chunks)
