

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Configuration Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FALLBACK_RESPONSE = "I couldn't find that in the uploaded documents."


STRICT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document assistant. Answer questions using ONLY the numbered context passages provided below.\n"
        "- Cite the source passages inline using bracketed numbers (e.g., [1], [2]).\n"
        f"- If the answer cannot be found within the context, respond explicitly with: \"{FALLBACK_RESPONSE}\"\n"
        "- Strictly avoid outside knowledge, assumptions, or extrapolation.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

LOOSE_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system", 
        "Answer the question using the provided context below:\n{context}"
    ),
    ("human", "{question}"),
])

QUERY_CONDENSATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the conversation history, rewrite the user's latest query into a standalone "
        "question that resolves any pronouns or ambiguous references. Return only the rewritten question."
    ),
    ("human", "Chat history:\n{history}\n\nLatest message: {question}"),
])


def initialize_llm() -> Any:
        return ChatOpenRouter(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            temperature=0.0
        )

def load_source_document(file_path: Union[str, Path], display_name: Optional[str] = None) -> List[Any]:
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .pdf or .docx.")
        
    documents = loader.load()
    source_identifier = display_name or path.name
    
    for doc in documents:
        doc.metadata["source"] = source_identifier
        
    return documents


def format_citation_label(document: Any) -> str:

    page_number = document.metadata.get("page")
    source_name = document.metadata.get("source", "document")
    
    if page_number is not None:
        return f"{source_name} (p.{page_number + 1})"
    return source_name


class DocumentQAEngine:
    
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150) -> None:
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)
        self.llm = initialize_llm()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.vectorstore: Optional[FAISS] = None

    def add_documents(self, file_registry: List[Tuple[Union[str, Path], Optional[str]]]) -> int:
        
        raw_documents = []
        for file_path, display_name in file_registry:
            raw_documents.extend(load_source_document(file_path, display_name))
            
        chunks = self.text_splitter.split_documents(raw_documents)
        if not chunks:
            raise ValueError("No extractable text found in documents. Note: Scanned PDFs require OCR preprocessing.")
            
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
            
        return len(chunks)

    def retrieve_context(self, query: str, candidate_k: int = 12, top_n: int = 4, use_reranker: bool = True) -> List[Any]:
       
        if not self.vectorstore:
            return []
            
        initial_k = candidate_k if use_reranker else top_n
        candidates = self.vectorstore.similarity_search(query, k=initial_k)
        
        if not use_reranker or not candidates:
            return candidates[:top_n]
            
        scoring_pairs = [(query, doc.page_content) for doc in candidates]
        relevance_scores = self.reranker.predict(scoring_pairs)
        
        ranked_results = sorted(
            zip(candidates, relevance_scores), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        return [doc for doc, _ in ranked_results[:top_n]]

    def _synthesize_standalone_query(self, query: str, chat_history: Optional[List[Tuple[str, str]]]) -> str:

        if not chat_history:
            return query
            
        formatted_history = "\n".join(f"{role}: {message}" for role, message in chat_history[-6:])
        chain = QUERY_CONDENSATION_PROMPT | self.llm | StrOutputParser()
        
        return chain.invoke({"history": formatted_history, "question": query}).strip()

    def generate_answer(
        self, 
        query: str, 
        chat_history: Optional[List[Tuple[str, str]]] = None, 
        use_baseline_mode: bool = False
    ) -> Tuple[str, List[Any]]:
      
        if self.vectorstore is None:
            return "Please upload and process documents before asking questions.", []
            
        standalone_query = self._synthesize_standalone_query(query, chat_history)
        retrieved_docs = self.retrieve_context(standalone_query, use_reranker=not use_baseline_mode)
        
        context_block = "\n\n".join(
            f"[{idx}] ({format_citation_label(doc)})\n{doc.page_content}" 
            for idx, doc in enumerate(retrieved_docs, start=1)
        )
        
        active_prompt = LOOSE_RAG_PROMPT if use_baseline_mode else STRICT_RAG_PROMPT
        generation_chain = active_prompt | self.llm | StrOutputParser()
        
        response = generation_chain.invoke({
            "context": context_block, 
            "question": standalone_query
        })
        
        return response, retrieved_docs
