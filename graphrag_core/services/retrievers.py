"""Retriever 構築ヘルパ（ui/state.py から移設、st非依存）。"""
from __future__ import annotations

# ParentDocumentRetriever version shim (community -> langchain fallback)
try:
    from langchain_community.retrievers.parent_document import ParentDocumentRetriever
    HAS_PARENT = True
except ImportError:
    try:
        from langchain.retrievers.parent_document import ParentDocumentRetriever
        HAS_PARENT = True
    except ImportError:
        HAS_PARENT = False


def create_vector_retriever(vector_store, top_k: int):
    """バージョン差異を吸収して Retriever を構築する。"""
    if vector_store is None:
        return None
    if HAS_PARENT:
        try:
            return ParentDocumentRetriever(
                vectorstore=vector_store, search_kwargs={"k": top_k},
            )
        except Exception as e:
            print(f"[Retriever] ParentDocumentRetriever unavailable. "
                  f"fallback=vector_store.as_retriever ({e})")
    return vector_store.as_retriever(search_kwargs={"k": top_k})
