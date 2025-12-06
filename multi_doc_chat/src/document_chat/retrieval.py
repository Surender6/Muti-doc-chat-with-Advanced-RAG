import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc", model_loader=loader)
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """
    
    def __init__(self, session_id: Optional[str], model_loader, retriever=None):
        try:
            self.session_id = session_id
            self.model_loader = model_loader

            # Load LLM ONE TIME using ModelLoader
            self.llm = self.model_loader.load_llm()

            # Load prompts
            self.contextuallize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy components
            self.retriever = retriever
            self.chain = None

            # If retriever exists, build chain automatically
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)

        except Exception as e:
            log.error("Failed to initialize Conversational RAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

        
    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
    
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )

            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}

            answer = self.chain.invoke(payload)

            # ------------------------------
            # FALLBACK: No context → answer normally
            # ------------------------------
            if answer is None or str(answer).strip().lower() in [
                "i don't know", "i don't know.", "i dont know", "i dont know."
            ]:
                fallback = self.llm.invoke(
                    f"Respond conversationally to the user: {user_input}"
                )
                answer = fallback.content
            # ------------------------------

            # No answer case
            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."

            # Validate answer using Pydantic
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)

            # Success log
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )

            return answer

        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

        
        # ---------- Internals ----------
        
    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)
        
    @staticmethod
    def _format_docs(docs):
        """Return None when no documents are retrieved (enables fallback)."""
        if not docs:
            return None
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    
    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)
            # 1) Rewrite user question with chat history context
            question_rewriter=(
                {"input":itemgetter("input"),"chat_history":itemgetter("chat_history")}
                | self.contextuallize_prompt
                | self.llm
                | StrOutputParser()
                
            )
            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs
            
            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
            {
                "context": retrieve_docs,
                "input" : itemgetter("input"),
                "chat_history": itemgetter("chat_history"),  
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
            )
            
            log.info("LCEL graph built successfully",session_id=self.session_id)
            
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
                          

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "similarity",
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ):
        """
        Load FAISS index from disk, build retriever, and rebuild LCEL chain.
        Supports MMR and similarity search.
        """
        try:
            embeddings = self.model_loader.load_embeddings()

            # Load FAISS vector store
            vs = FAISS.load_local(
                index_path,
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True
            )

            # Configure retriever
            if search_type == "mmr":
                self.retriever = vs.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,
                        "fetch_k": fetch_k,
                        "lambda_mult": lambda_mult,
                    }
                )
                log.info(
                    "using mmr search",
                    k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
                )
            else:
                self.retriever = vs.as_retriever(search_kwargs={"k": k})

            # Rebuild LCEL chain with updated retriever
            self._build_lcel_chain()

            log.info(
                "FAISS retriever loaded successfully",
                index=index_path,
                k=k,
                search_type=search_type,
                session_id=self.session_id
            )

        except Exception as e:
            log.error("Failed to load retriever from FAISS", error=str(e))
            raise DocumentPortalException("Failed to load retriever from FAISS", sys)
    

    
            
            
            
                    
        
                

