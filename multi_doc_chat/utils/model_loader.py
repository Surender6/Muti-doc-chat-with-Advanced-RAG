import os
import sys
import json
import yaml
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.utils.config_loader import load_config
#from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from langchain_community.embeddings import HuggingFaceEmbeddings

class ApiKeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY"]

    def __init__(self):
        self.api_keys = {}

        # Try to load raw JSON string from environment
        raw = os.getenv("apikeyliveclass")   # e.g. {"GROQ_API_KEY": "..."}
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("apikeyliveclass is not a valid JSON object")
                self.api_keys = parsed
                log.info("Loaded GROQ_API_KEY from JSON secret")
            except Exception as e:
                log.warning("Failed to parse JSON API key", error=str(e))

        # Fallback: try loading from individual environment variable
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from environment variable")

        # Final validation check
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API key", missing_keys=missing)
            raise DocumentPortalException("Missing required API key", sys)

        # Mask sensitive output
        log.info(
            "API key loaded successfully",
            keys={k: v[:6] + "..." for k, v in self.api_keys.items()}
        )

    def get(self, key: str) -> str:
        val = self.api_keys.get(key)
        if not val:
            raise KeyError(f"API key `{key}` is missing")
        return val

        
         
class ModelLoader:
    def __init__(self, config_path: str = None):
        try:
            # Auto-locate config.yaml inside multi_doc_chat/config/
            if config_path is None:
                base_dir = Path(__file__).resolve().parents[1]   # multi_doc_chat folder
                config_path = base_dir / "config" / "config.yaml"

            # Load YAML config
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

            # Initialize API key manager
            self.api_key_mgr = ApiKeyManager()

            log.info("ModelLoader initialized", config_path=str(config_path))

        except Exception as e:
            log.error("ModelLoader initialization failed", error=str(e))
            raise DocumentPortalException("ModelLoader initialization failed", sys)
        
    def load_embeddings(self):
        """
       Load and return embedding model (HuggingFace or your chosen embedding model).
       Using GROQ only for LLM, so embeddings come from HF.
       """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            # Use HF embeddings (common & compatible with FAISS + RAG)
            return HuggingFaceEmbeddings(model_name=model_name)

        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)
            
           
    def load_llm(self):
        """
        Load and return the GROQ LLM (only provider).
        """
        try:
           
           llm_config = self.config["llm"]
           provider = llm_config.get("provider")
           model_name= llm_config.get("model_name")
           temperature = llm_config.get("temperature",0.2)
           max_tokens = llm_config.get("max_output_tokens", 2048)
           
           log.info("Loading LLM", provider=provider, model=model_name)
           
           if provider != "groq":
               raise ValueError("only 'groq' provider is supported in this project")
           
           return ChatGroq(
               model= model_name,
               api_key=self.api_key_mgr.get("GROQ_API_KEY"),
               temperature=temperature,
               max_tokens=max_tokens,
           )
        except Exception as e:
           log.error("Error loading LLM", error=str(e))
           raise DocumentPortalException("Failed to load LLM", sys)
                   
if __name__ == "main":
    loader = ModelLoader()
    
   # ---- Test Embeddings ----
    try:
        embeddings = loader.load_embeddings()
        print(f"Embedding Model Loaded: {embeddings}")

        result = embeddings.embed_query("Hello, how are you?")
        print(f"Embedding Result Length: {len(result)}")
        print(f"First 5 Embedding Values: {result[:5]}")

    except Exception as e:
        print("Embedding test failed:", e)
    
    # ---- Test LLM ----
    try:
        llm = loader.load_llm()
        print(f"LLM Loaded: {llm}")

        response = llm.invoke("Hello, how are you?")
        print(f"LLM Result: {response}")  # ChatGroq returns string, not .content

    except Exception as e:
        print("LLM test failed:", e)