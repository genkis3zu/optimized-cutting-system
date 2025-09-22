# 鋼板切断最適化システム / Steel Cutting Optimization System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ギロチンカット制約下での2Dビンパッキング最適化による鋼板切断システム。材料効率向上と作業時間短縮を実現します。

*A steel cutting optimization system using 2D bin packing with guillotine cut constraints to improve material efficiency and reduce working time.*

## 🎯 システム概要 / System Overview

本システムは製造現場での実際の鋼板切断作業を効率化するため、以下の機能を提供します：

- **材料在庫管理**: 実際の在庫データ（sizaidata.txt）との連携
- **切断最適化**: 複数アルゴリズムによる効率的なパネル配置
- **作業指示生成**: 実際の切断作業に対応した詳細な指示書
- **インタラクティブ可視化**: 切断レイアウトのリアルタイム表示
- **ERP/MES統合**: 外部システムとの連携機能

## 🚀 主要機能 / Key Features

### ✅ 実装済み機能 / Implemented Features

#### 🏭 材料在庫管理システム
- **永続化対応**: JSON形式での在庫データ保存
- **自動データ読み込み**: sizaidata.txtからの材料情報自動インポート
- **材料コード正規化**: KW300→KW-300等の自動変換
- **リアルタイム検証**: パネル要求に対する在庫チェック

#### ⚡ 最適化エンジン
- **複数アルゴリズム対応**:
  - First Fit Decreasing (FFD) - 高速処理
  - Best Fit Decreasing (BFD) - 高効率
  - Hybrid Algorithm - バランス型
- **ギロチンカット制約**: 実際の切断機制約に対応
- **材質別処理**: 材料タイプごとのブロック化処理

#### 🎨 インタラクティブUI
- **Streamlit多ページ設計**: 最適化/材料管理の分離
- **リアルタイム可視化**: Plotlyによる切断レイアウト表示
- **日英対応インターフェース**: バイリンガル対応

#### 📊 可視化・レポート機能
- **切断レイアウト図**: 色分けされたパネル配置
- **効率メトリクス**: 材料効率・コスト計算
- **作業指示書**: PDF/Excel形式での出力
- **品質チェックポイント**: 製造品質管理

#### 🔗 統合・連携機能
- **FastAPI REST API**: 外部システム連携
- **ERP/MES コネクタ**: 生産管理システム統合
- **複数データ形式対応**: CSV, TSV, JSON, 固定幅

### 🏗️ アーキテクチャ / Architecture

```
steel-cutting-system/
├── 📱 app.py                     # メインアプリケーション
├── 🧠 core/                      # コアロジック
│   ├── models.py                 # データ構造定義
│   ├── optimizer.py              # 最適化エンジン
│   ├── material_manager.py       # 材料在庫管理 ⭐新機能
│   ├── text_parser.py            # 堅牢テキストパーサー
│   └── algorithms/               # 最適化アルゴリズム
│       ├── ffd.py               # First Fit Decreasing
│       ├── bfd.py               # Best Fit Decreasing
│       └── hybrid.py            # ハイブリッドアルゴリズム
├── ✂️ cutting/                   # 切断作業機能
│   ├── instruction.py           # 作業指示生成
│   ├── sequence.py              # 切断順序最適化
│   ├── validator.py             # 制約検証
│   ├── quality.py               # 品質管理
│   └── export.py                # レポート出力
├── 🎨 ui/                        # ユーザーインターフェース
│   ├── components.py            # UIコンポーネント
│   ├── visualizer.py            # 可視化機能 ⭐新機能
│   └── material_management_ui.py # 材料管理UI ⭐新機能
├── 🔌 integration/               # システム統合
│   ├── api.py                   # REST APIエンドポイント
│   └── erp_connector.py         # ERP/MES連携
├── 📊 config/                    # 設定・データ
│   └── material_inventory.json  # 材料在庫データ ⭐新機能
└── 🧪 tests/                     # テストスイート
    ├── unit/                    # ユニットテスト
    ├── integration/             # 統合テスト
    └── performance/             # 性能テスト
```

## 🛠️ セットアップ / Setup

### 前提条件 / Prerequisites
- Python 3.9+
- 推奨メモリ: 8GB+
- ブラウザ: Chrome/Edge（最新版）

### インストール / Installation

```bash
# リポジトリのクローン
git clone https://github.com/genkis3zu/optimized-cutting-system.git
cd optimized-cutting-system

# 仮想環境の作成・有効化
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 依存関係のインストール
pip install -r requirements.txt
```

### 実行 / Running

```bash
# Streamlitアプリケーションの起動
streamlit run app.py

# カスタムポートでの起動
streamlit run app.py --server.port 8080

# 開発モード（自動リロード）
streamlit run app.py --server.runOnSave true
```

アプリケーションは `http://localhost:8501` で利用可能になります。

## 📖 使用方法 / Usage

### 1. 材料在庫の設定
1. **材料管理ページ**に移動
2. **サンプルデータの読み込み**または**手動での材料追加**
3. 材料コード、寸法、在庫数の確認・編集

### 2. 切断最適化の実行
1. **最適化ページ**でパネル情報を入力
   - サンプルデータ（data0923.txt）の利用
   - 手動入力フォームでの個別追加
2. **材料検証**結果の確認
3. **最適化実行**ボタンでアルゴリズム実行
4. **結果の確認**：
   - 切断レイアウト図
   - 効率メトリクス
   - 材料使用状況

### 3. 作業指示・レポート出力
- **作業指示書生成**: 実際の切断手順書
- **レポート出力**: Excel形式での詳細分析
- **結果保存**: セッション状態での保存

## ⚙️ 設定とカスタマイズ / Configuration

### アルゴリズム選択
```python
# 自動選択（推奨）
algorithm = "AUTO"

# 手動選択
algorithm = "FFD"    # 高速処理（<1秒）
algorithm = "BFD"    # 高効率（~5秒）
algorithm = "HYBRID" # バランス型（~15秒）
```

### 制約条件の設定
```python
constraints = {
    'max_sheet_width': 1500,      # 最大シート幅（mm）
    'max_sheet_height': 3100,     # 最大シート高さ（mm）
    'min_panel_size': 50,         # 最小パネルサイズ（mm）
    'kerf_width': 3.5,            # 切断代（mm）
    'allow_rotation': True        # パネル回転許可
}
```

## 📊 パフォーマンス / Performance

### 処理時間目安
| パネル数 | アルゴリズム | 処理時間 | 期待効率 |
|---------|------------|---------|----------|
| ≤10     | FFD        | <1秒    | 70-75%   |
| ≤30     | BFD        | <5秒    | 80-85%   |
| ≤50     | HYBRID     | <15秒   | 85%+     |
| ≤100    | HYBRID     | <30秒   | ベストエフォート |

### メモリ使用量
- 通常動作: ~100-200MB
- 大規模データ: <512MB
- 最大許容: 1GB

## 🧪 テスト / Testing

```bash
# 全テストの実行
python -m pytest tests/

# カバレッジ付きテスト
python -m pytest tests/ --cov=core --cov=cutting --cov-report=html

# 特定テストの実行
python -m pytest tests/unit/test_material_manager.py -v

# 性能テスト
python -m pytest tests/performance/ -v
```

## 🔧 開発・保守 / Development

### コード品質
```bash
# リンティング
python -m pylint core/ cutting/ ui/

# フォーマッティング
python -m black core/ cutting/ ui/ tests/

# 型チェック
python -m mypy core/ cutting/ ui/
```

### API開発
```bash
# FastAPI開発サーバー起動
uvicorn integration.api:app --reload --port 8000

# API仕様確認
curl http://localhost:8000/docs
```

## 📝 ドキュメント / Documentation

- [📋 **プロジェクト仕様書**](steel_cutting_spec.md) - 詳細な技術仕様
- [⚙️ **開発ガイド**](CLAUDE.md) - Claude Code向け開発指針
- [🏗️ **アーキテクチャ概要**](docs/architecture.md) - システム設計詳細
- [📚 **API リファレンス**](docs/api_reference.md) - REST API仕様

## 🤝 貢献 / Contributing

1. Forkしてください
2. フィーチャーブランチを作成: `git checkout -b feature/amazing-feature`
3. 変更をコミット: `git commit -m 'Add amazing feature'`
4. ブランチにプッシュ: `git push origin feature/amazing-feature`
5. Pull Requestを作成

## 📄 ライセンス / License

このプロジェクトはMITライセンスの下で公開されています。詳細は [LICENSE](LICENSE) をご覧ください。

## 🆘 サポート / Support

- **Issue報告**: [GitHub Issues](https://github.com/genkis3zu/optimized-cutting-system/issues)
- **機能要求**: [Feature Requests](https://github.com/genkis3zu/optimized-cutting-system/discussions)
- **ドキュメント**: [Wiki](https://github.com/genkis3zu/optimized-cutting-system/wiki)

## 📊 プロジェクト状況 / Project Status

### ✅ 完了済みフェーズ / Completed Phases
- **Phase 1**: コア機能開発（データ構造、基本アルゴリズム、UI）
- **Phase 2**: 最適化強化（複数アルゴリズム、材料管理、可視化）
- **Phase 3**: 作業指示・統合機能（切断指示、品質管理、API）

### 🚧 現在の開発状況 / Current Development Status
- **材料在庫管理**: ✅ 完全実装
- **切断最適化**: ✅ 複数アルゴリズム対応
- **インタラクティブUI**: ✅ 多ページ設計
- **可視化機能**: ✅ Plotly統合
- **作業指示生成**: ✅ PDF/Excel出力
- **ERP/MES統合**: ✅ REST API実装
- **品質管理**: ✅ チェックポイント管理

### 🔮 次期バージョン予定 / Future Roadmap
- **機械学習最適化**: 過去データからの学習
- **リアルタイム在庫連携**: 自動在庫更新
- **モバイル対応**: タブレット・スマートフォン最適化
- **多言語対応**: 英語・中国語・韓国語

---

**最終更新**: 2025年1月 | **バージョン**: v1.0.0