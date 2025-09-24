"""
Common Styles for Steel Cutting Optimization System
鋼板切断最適化システム共通スタイル

Unified design system with consistent colors, typography, and components
"""

# カラーパレット / Color Palette
PRIMARY_COLOR = "#1f77b4"      # メインブルー
SUCCESS_COLOR = "#2ca02c"      # グリーン
WARNING_COLOR = "#ff7f0e"      # オレンジ
ERROR_COLOR = "#d62728"        # レッド
NEUTRAL_COLOR = "#7f7f7f"      # グレー
BACKGROUND_COLOR = "#f8f9fa"   # 背景グレー

def get_common_css():
    """共通CSSスタイルを返す"""
    return """
    <style>
    /* 共通ヘッダースタイル */
    .unified-header {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SUCCESS_COLOR} 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(31, 119, 180, 0.3);
    }}

    .unified-header h1 {{
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }}

    .unified-header p {{
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }}

    /* 統一メトリックカード */
    .metric-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {PRIMARY_COLOR};
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }}

    .metric-card h3 {{
        margin: 0 0 0.5rem 0;
        font-size: 1rem;
        color: {NEUTRAL_COLOR};
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .metric-card h2 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: {PRIMARY_COLOR};
    }}

    .metric-card p {{
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        color: {NEUTRAL_COLOR};
    }}

    /* コンテンツセクション */
    .content-section {{
        background: white;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}

    .content-section h2 {{
        margin: 0 0 1rem 0;
        color: {PRIMARY_COLOR};
        border-bottom: 2px solid {PRIMARY_COLOR};
        padding-bottom: 0.5rem;
    }}

    /* アクションカード */
    .action-card {{
        background: {BACKGROUND_COLOR};
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid {WARNING_COLOR};
        margin: 1rem 0;
    }}

    /* ステータスメッセージ */
    .status-success {{
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {SUCCESS_COLOR};
        margin: 1rem 0;
    }}

    .status-warning {{
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {WARNING_COLOR};
        margin: 1rem 0;
    }}

    .status-error {{
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {ERROR_COLOR};
        margin: 1rem 0;
    }}

    /* テーブルスタイル */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    /* ボタンスタイル調整 */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}

    /* フォームスタイル */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {{
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.2s;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {{
        border-color: {PRIMARY_COLOR};
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.1);
    }}

    /* チャートコンテナ */
    .chart-container {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }}

    /* レスポンシブ調整 */
    @media (max-width: 768px) {{
        .unified-header {{
            padding: 1.5rem;
        }}

        .unified-header h1 {{
            font-size: 2rem;
        }}

        .metric-card {{
            padding: 1rem;
        }}

        .content-section {{
            padding: 1.5rem;
        }}
    }}
    </style>
    """

def get_metric_colors():
    """メトリック表示用のカラーセットを返す"""
    return {
        'primary': PRIMARY_COLOR,
        'success': SUCCESS_COLOR,
        'warning': WARNING_COLOR,
        'error': ERROR_COLOR,
        'neutral': NEUTRAL_COLOR
    }

def get_gradient_colors():
    """グラデーション用のカラーペアを返す"""
    return [
        (PRIMARY_COLOR, SUCCESS_COLOR),    # ブルー→グリーン
        (WARNING_COLOR, ERROR_COLOR),      # オレンジ→レッド
        (SUCCESS_COLOR, PRIMARY_COLOR),    # グリーン→ブルー
        (NEUTRAL_COLOR, PRIMARY_COLOR)     # グレー→ブルー
    ]
