import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from company_policy_chat.utils.config_loader import load_config

log = logging.getLogger(__name__)

class ApiKeyManager:
    
    def __init__(self):
        self._keys: Dict[str, str] = {}
        self._load_keys()

    def _load_keys(self):
        
        required_keys = ['GROQ_API_KEY']
        for key in required_keys:
            value = os.getenv(key)
            if not value:
                
                log.error(f"Missing required environment variable: {key}")
                raise KeyError(f"API Key '{key}' not found. Check your .env file.")
            self._keys[key] = value

    def get(self, key: str) -> str:
        return self._keys.get(key, "")

class ModelLoader:
    
    def __init__(self):
        load_dotenv()  
        self.config = load_config()
        self.api_keys = ApiKeyManager()

    def load_embedding(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info(f"Loading embedding model: {model_name}")
            return FastEmbedEmbeddings(model=model_name)
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            raise

    def load_llm(self):
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "groq")
        
        spec = llm_cfg.get(provider, {})
        model_name = spec.get("model_name")
        temp = spec.get("temperature", 0.2)

        log.info(f"Initializing LLM: {provider} | Model: {model_name}")

        if provider == "groq":
            return self._init_groq(model_name, temp)
        
        raise ValueError(f"Unsupported provider: {provider}")

    def _init_groq(self, model: str, temperature: float):
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model,
                temperature=temperature,
                api_key=self.api_keys.get("GROQ_API_KEY")
            )
        except ImportError:
            log.error("langchain-groq package not found.")
            raise

if __name__ == "__main__":

    loader = ModelLoader()
    
    try:
        embeddings = loader.load_embedding()
        llm = loader.load_llm()
        print("sample embedding: ",embeddings.embed_query("how are you?"))
        print("Models loaded successfully.")
        response = llm.invoke("how are you?")
        print(response.content)
    except Exception as err:
        print(f"Initialization failed: {err}")