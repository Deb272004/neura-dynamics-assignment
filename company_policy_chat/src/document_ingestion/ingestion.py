from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from company_policy_chat.utils.model_loader import ModelLoader
from company_policy_chat.logger.custom_logger import CustomLogger
from company_policy_chat.utils.config_loader import load_config
from company_policy_chat.utils.file_utils import (
    save_uploaded_files,
    load_documents,
)



config = load_config()
log = CustomLogger().get_logger(__name__)


class FaissManager:
    """Handles creation, loading, and incremental updates of FAISS index."""

    def __init__(self, index_dir: Path, model_loader: ModelLoader):
        self.index_dir = index_dir
        self.embeddings = model_loader.load_embedding()
        self.vs: FAISS | None = None

    @property
    def index_path(self) -> Path:
        return self.index_dir / "index.faiss"

    def load_or_create(self, docs: List[Document]) -> FAISS:
        """Load an existing FAISS index or create/update it with new documents."""

        if self.index_path.exists():
            log.info("Loading existing FAISS index", path=str(self.index_dir))

            self.vs = FAISS.load_local(
                folder_path=str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )

            self._add_new_documents(docs)

        else:
            log.info("Creating new FAISS index", path=str(self.index_dir))

            self.vs = FAISS.from_documents(docs, self.embeddings)
            self.vs.save_local(str(self.index_dir))

        return self.vs

    def _add_new_documents(self, docs: List[Document]) -> None:
        
        existing_sources = set()

        if self.vs and self.vs.docstore:
            for doc in self.vs.docstore._dict.values():
                source = doc.metadata.get("source")
                if source:
                    existing_sources.add(source)
                    

        new_docs = [
            doc for doc in docs
            if doc.metadata.get("source") not in existing_sources
        ]

        if not new_docs:
            log.info("No new documents to add — index is up to date")
            return

        log.info("Adding new documents to FAISS", count=len(new_docs))
        self.vs.add_documents(new_docs)
        self.vs.save_local(str(self.index_dir))

class Ingestion:
    

    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
    ):
        try:
            self.model_loader = ModelLoader()

            self.temp_base = Path(temp_base).resolve()
            self.temp_base.mkdir(parents=True, exist_ok=True)

            self.faiss_base = Path(faiss_base).resolve()
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            log.info(
                "Ingestion initialized",
                temp_dir=str(self.temp_base),
                faiss_dir=str(self.faiss_base),
            )

        except Exception as e:
            log.error("Failed to initialize Ingestion", error=str(e))
            raise RuntimeError("Ingestion initialization failed") from e

    def _split_documents(
        self,
        docs: List[Document],
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunks = splitter.split_documents(docs)

        log.info(
            "Documents split into chunks",
            chunks=len(chunks),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return chunks

    def build_index(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> FAISS:
       

        try:
            
            paths = save_uploaded_files(uploaded_files, self.temp_base)

            
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents could be loaded")

            
            chunks = self._split_documents(
                docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            
            faiss_manager = FaissManager(
                index_dir=self.faiss_base,
                model_loader=self.model_loader,
            )

            vs = faiss_manager.load_or_create(chunks)

            log.info(
                "FAISS index ready",
                total_chunks=len(chunks),
                index_path=str(self.faiss_base),
            )

            return vs

        except Exception as e:
            log.error("Failed to build FAISS index", error=str(e))
            raise
