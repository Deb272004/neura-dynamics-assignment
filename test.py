import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from company_policy_chat.src.document_ingestion.ingestion import Ingestion
from company_policy_chat.src.document_retrieval.retrieval import Retrieval

load_dotenv()


def test_ingestion_and_retrieval():
    try:
        
        test_assets_dir = Path(
            "/home/deblina/Documents/projects/neura dynamics assignment/test_assets"
        )
        uploaded_files = []

        for file_path in test_assets_dir.glob("*"):  
            if file_path.is_file():
                uploaded_files.append(open(file_path, "rb"))

        if not uploaded_files:
            print(f" No test files found in {test_assets_dir}")
            sys.exit(1)

        print("\n--- Starting Ingestion ---")
        print(f"Found {len(uploaded_files)} files for ingestion:")
        for f in uploaded_files:
            print(f" - {Path(f.name).name}")

        
        ingestor = Ingestion(temp_base="data", faiss_base="faiss_index")
        ingestor.build_index(
            uploaded_files=uploaded_files,
            chunk_size=500,
            chunk_overlap=50,
        )

        
        data_dir = Path("data")
        saved_files = list(data_dir.glob("*"))

        if not saved_files:
            print("No files persisted during ingestion")
            sys.exit(1)

        print(f" Files persisted: {[f.name for f in saved_files]}")

        
        faiss_dir = Path("faiss_index")
        index_file = faiss_dir / "index.faiss"

        if not index_file.exists():
            print(" FAISS index missing")
            sys.exit(1)

        print(f"✅ FAISS index created at {faiss_dir}")

        
        retriever = Retrieval()
        retriever.load_retriever_from_faiss(index_path="faiss_index")

        print("Retrieval class initialized and FAISS index loaded")

        
        chat_history = []
        answer = ""
        user_input = ""

        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                answer = "goodbye!"
                break

            
            answer = retriever.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        
        print("Retrieval test passed")
        print("\nIngestion + Retrieval Test PASSED")

    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        
        for f in uploaded_files:
            f.close()


if __name__ == "__main__":
    test_ingestion_and_retrieval()
