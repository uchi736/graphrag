"""graphrag_core.document - ドキュメント処理

軽量 import のみ eager。Azure DI (重い) は lazy で、
`from graphrag_core.document.azure_di import ...` で明示呼び出し。
"""

from graphrag_core.document.pdf import extract_pdf_text, load_pdf_text
from graphrag_core.document.onprem_pdf import extract_pdf_onprem

__all__ = [
    "extract_pdf_text",
    "load_pdf_text",
    "extract_pdf_onprem",
]
