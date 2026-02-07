from __future__ import annotations
import os
import traceback
from operator import itemgetter
from typing import List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from company_policy_chat.utils.model_loader import ModelLoader
from company_policy_chat.logger.custom_logger import CustomLogger
from company_policy_chat.utils.config_loader import load_config
from company_policy_chat.prompts.prompts_library import CONTEXT_QA_PROMPT, CONTEXTUALIZE_QUESTION_PROMPT

config = load_config()
log = CustomLogger().get_logger(__name__)

class Retrieval:


    def __init__(self):
        self.chat_history: List[BaseMessage] = []
        self.chain = None
        self.retriever = None

        try:
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = CONTEXTUALIZE_QUESTION_PROMPT
            self.qa_prompt: ChatPromptTemplate = CONTEXT_QA_PROMPT
            log.info("Retrieval class initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize Retrieval class: {e}")
            raise

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = config["retriever"]["k"],
        search_type: str = config["retriever"]["search_type"],
        fetched_k: int = config["retriever"]["fetched_k"],
        index_name: str = "index",  
        lambda_mult: float = config["retriever"]["lambda_mult"],
    ):
        
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            if search_type not in {"similarity", "mmr"}:
                raise ValueError(f"Unsupported search_type: {search_type}")

            embeddings = ModelLoader().load_embedding()

            
            vectorstore = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )

            
            if search_type == "mmr":
                search_kwargs = {"fetch_k": fetched_k, "lambda_mult": lambda_mult, "k": k}
            else:
                search_kwargs = {"k": k}

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs
            )

            self._build_lcel_chain()
            log.info("FAISS retriever loaded successfully")

        except Exception as e:
            log.error(f"Failed to load retriever from FAISS: {e}")
            traceback.print_exc()
            raise

        
    

    def invoke(self, user_input: str, chat_history:BaseMessage) -> str:
        
        try:
            if self.chain is None:
                raise ValueError("LCEL chain not initialized. Call load_retriever_from_faiss() first.")

           
            payload = {
                "input": user_input,
                "chat_history": chat_history
            }

            answer = self.chain.invoke(payload)

            if not answer:
                log.warning("Answer not generated")
                return "Answer not generated."

            return answer

        except Exception as e:
            log.error(f"Failed to invoke retrieval: {e}")
            traceback.print_exc()
            raise


    def _load_llm(self):
        
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")

            log.info("LLM loaded successfully")
            return llm

        except Exception as e:
            log.error(f"Failed to load LLM: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def _format_docs(docs) -> str:
       
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        
        try:
            if self.retriever is None:
                raise ValueError("Retriever must be set before building LCEL chain")

            
            question_rewriter = (
                {
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            
            retrieved_docs = (
                question_rewriter
                | self.retriever
                | Retrieval._format_docs
            )

            
            self.chain = (
                {
                    "context": retrieved_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL chain built successfully")

        except Exception as e:
            log.error(f"Failed to build LCEL chain: {e}")
            traceback.print_exc()
            raise
