"""
RAG Pipeline using LangChain with FAISS Vector Store.

This module provides a complete RAG implementation:
  - Document loading from txt files
  - Text chunking with configurable parameters
  - Embedding generation using OpenAI
  - FAISS vector store for similarity search
  - Retrieval-augmented generation

Usage:
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    rag.load_documents("data")
    result = rag.query("What is the PTO policy?")
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# RAG Pipeline
# =============================================================================


class RAGPipeline:
    """A configurable RAG pipeline for question answering over documents."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 4,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            model_name: OpenAI model for generation.
            embedding_model: OpenAI model for embeddings.
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks.
            top_k: Number of documents to retrieve.
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize components
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # State
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.chain = None
        self.documents: list[Document] = []

    def load_documents(self, data_dir: str) -> int:
        """
        Load all .txt files from a directory.

        Args:
            data_dir: Path to directory containing txt files.

        Returns:
            Number of chunks created.

        Raises:
            FileNotFoundError: If directory doesn't exist.
            ValueError: If no txt files are found.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # Load all txt files
        loader = DirectoryLoader(
            str(data_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        raw_documents = loader.load()

        if not raw_documents:
            raise ValueError(f"No .txt files found in {data_dir}")

        # Split into chunks
        self.documents = self.text_splitter.split_documents(raw_documents)

        # Create vector store
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # Build RAG chain
        self._build_chain()

        print(f"Loaded {len(raw_documents)} files, created {len(self.documents)} chunks")
        return len(self.documents)

    def _build_chain(self) -> None:
        """Build the RAG chain with retriever and LLM."""
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant answering questions based on the provided context.
Use ONLY the information from the context to answer. If the context doesn't contain
enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
        )

        def format_docs(docs: list[Document]) -> str:
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        """
        Query the RAG pipeline.

        Args:
            question: The question to answer.

        Returns:
            Dictionary with keys:
                - input: Original question
                - output: Generated answer
                - context: List of retrieved document contents
                - sources: List of source file names

        Raises:
            RuntimeError: If pipeline not initialized.
        """
        if self.chain is None:
            raise RuntimeError("Pipeline not initialized. Call load_documents() first.")

        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)

        # Generate answer
        answer = self.chain.invoke(question)

        # Extract context and sources
        context = [doc.page_content for doc in retrieved_docs]
        sources = list(set(
            Path(doc.metadata.get("source", "unknown")).name
            for doc in retrieved_docs
        ))

        return {
            "input": question,
            "output": answer,
            "context": context,
            "sources": sources,
        }

    def retrieve(self, question: str) -> list[Document]:
        """
        Retrieve relevant documents without generating an answer.

        Args:
            question: The query.

        Returns:
            List of retrieved Document objects.
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not initialized. Call load_documents() first.")
        return self.retriever.invoke(question)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    print("Initializing RAG pipeline...")
    rag = RAGPipeline()
    rag.load_documents(str(data_dir))

    demo_questions = [
        "How many days of PTO do employees accrue per month?",
        "What are the rate limits for the API?",
        "How do I reset my password in Acme Notes?",
        "What was the resolution for TICKET-1003?",
        "What details should I collect for a sync issue?",
    ]

    for question in demo_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        result = rag.query(question)
        print(f"A: {result['output']}")
        print(f"Sources: {result['sources']}")
