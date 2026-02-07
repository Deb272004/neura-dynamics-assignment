from pathlib import Path
from typing import Iterable, List, BinaryIO
import shutil

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
    ToMarkdownLoader
)

from company_policy_chat.logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".md"}




def _get_filename(file_obj) -> str:
    """Extract filename from UploadFile or file-like object."""
    if hasattr(file_obj, "filename") and file_obj.filename:
        return Path(file_obj.filename).name

    if hasattr(file_obj, "name"):
        return Path(file_obj.name).name

    return "unknown_file"


def _get_file_stream(file_obj) -> BinaryIO:
    """Extract readable binary stream."""
    if hasattr(file_obj, "file"):  
        return file_obj.file
    return file_obj                 


def save_uploaded_files(uploaded_files, target_dir: Path) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for file_obj in uploaded_files:
        filename = _get_filename(file_obj)
        out_path = target_dir / filename

        try:
            stream = _get_file_stream(file_obj)

           
            try:
                stream.seek(0)
            except Exception:
                pass

            data = stream.read()
            if not data:
                log.error("Empty file input, skipping", file=filename)
                continue

            with open(out_path, "wb") as f:
                f.write(data)

            saved.append(out_path)
            log.info("File saved", file=filename, size=len(data))

        except Exception as e:
            log.error("Failed saving file", file=filename, error=str(e))

    return saved


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load documents using the appropriate LangChain loader.
    """
    documents: List[Document] = []

    for path in paths:
        try:
            if not path.exists():
                log.warning("File does not exist", path=str(path))
                continue

            if path.stat().st_size == 0:
                log.error("Skipping empty file", path=str(path))
                continue

            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported extension skipped", path=str(path))
                continue

            loader = _get_loader(path)
            docs = loader.load()

            if not docs:
                log.warning("No documents extracted", path=str(path))
                continue

            documents.extend(docs)

        except Exception as e:
            log.error(
                "Failed loading document",
                path=str(path),
                error=str(e),
            )

    log.info("Documents loaded", count=len(documents))
    return documents



def _get_loader(path: Path):
    """Return correct LangChain loader for a file."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    if ext == ".txt":
        return TextLoader(str(path))
    if ext == ".csv":
        return CSVLoader(str(path))
    if ext == ".md":
        return CSVLoader(str(path))

    raise ValueError(f"Unsupported extension: {ext}")
