import os
import io
import types
import json
import shutil
import pathlib
import sys
import pytest

os.environ.setdefault("PYTHONPATH", str(pathlib.Path(__file__).resolve().parents[1] / "multi_doc_chat"))
os.environ.setdefault("GROQ_API_KEY", "dummy")

from fastapi.testclient import TestClient

# Ensure repository root is importable for import main

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import main

@pytest.fixture
def client():
    return TestClient(main.app)

@pytest.fixture
def clear_sessions():
    main.SESSIONS.clear()
    yield
    main.SESSIONS.clear()
    
    
@pytest.fixture
def temp_dirs(tmp_path: pathlib.Path):
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        # Point working directories used by app code to tmp ones by chdir
        os.chdir(tmp_path)
        yield {"data":data_dir,"faiss":faiss_dir}
    finally:
        os.chdir(cwd)
        
        
class _StubEmbeddings:
    def embed_query(self, text: str):
        return [0.0, 0.1, 0.2]
    
    def embed_documents(self,texts):
         return [[0.0, 0.1, 0.2] for _ in texts]
    
    def __call__(self, text: str):
        return [0.0, 0.1, 0.2]
    
    
class _StubLLM:
    def invoke(self, input):
        return "stubbed answer"
    

    

    
