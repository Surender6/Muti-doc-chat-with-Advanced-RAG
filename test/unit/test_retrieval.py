import pytest

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


def test_conversationalrag_invoke_raises_when_no_retriever(
    temp_dirs,
    stub_model_loader,
):
    model_loader = stub_model_loader()
    rag = ConversationalRAG(session_id="s1",model_loader=model_loader)

    with pytest.raises(DocumentPortalException):
        rag.invoke("hello")


def test_conversationalrag_load_retriever_invalid_path_raises(
    temp_dirs,
    stub_model_loader,
):
    model_loader = stub_model_loader()
    rag = ConversationalRAG(session_id="s1",model_loader = model_loader,
    )

    with pytest.raises(DocumentPortalException):
        rag.load_retriever_from_faiss(
            index_path="faiss_index/does_not_exist"
        )
