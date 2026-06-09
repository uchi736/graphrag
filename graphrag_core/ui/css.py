"""Streamlit カスタムCSS"""

CUSTOM_CSS = """<style>
/* ===== Google Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ===== 全体フォント ===== */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ===== メインタイトル ===== */
h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* ===== サイドバー ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.1);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #c0c0d0;
}

/* ===== プライマリボタン ===== */
button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}
button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.5);
    transform: translateY(-1px);
}

/* ===== セカンダリボタン ===== */
.stButton > button:not([kind="primary"]) {
    border-radius: 8px;
    border: 1px solid rgba(102, 126, 234, 0.3);
    transition: all 0.2s ease;
    font-weight: 500;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #667eea;
    background-color: rgba(102, 126, 234, 0.08);
}

/* ===== タブ ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid rgba(102, 126, 234, 0.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(102, 126, 234, 0.1);
    border-bottom: 3px solid #667eea;
    font-weight: 600;
}

/* ===== Expander (カード風) ===== */
.streamlit-expanderHeader {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
}
details[data-testid="stExpander"] {
    border: 1px solid rgba(102, 126, 234, 0.15);
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    transition: box-shadow 0.2s ease;
}
details[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.12);
}

/* ===== メトリクス (カード風) ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border: 1px solid rgba(102, 126, 234, 0.12);
    border-radius: 10px;
    padding: 16px;
}
[data-testid="stMetric"] label {
    font-weight: 600;
    color: #667eea;
}

/* ===== テキストエリア & テキスト入力 ===== */
.stTextArea textarea, .stTextInput input {
    border-radius: 8px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.15);
}

/* ===== 情報・警告・エラーメッセージ ===== */
.stAlert {
    border-radius: 8px;
}

/* ===== スピナー ===== */
.stSpinner > div {
    border-top-color: #667eea;
}

/* ===== セレクトボックス ===== */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* ===== プログレスバー ===== */
.stProgress > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* ===== スクロールバー (メインエリア) ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.3);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(102, 126, 234, 0.5);
}
</style>
"""
