# 鋼板切断最適化システム開発仕様書

## 1. システム概要

### 1.1 目的
鋼板切断作業において、ギロチンカット制約下でのパネル配置を自動最適化し、材料効率の向上と作業時間短縮を実現する。

### 1.2 対象業務
- 鋼板からの複数サイズパネル切り出し作業
- 特に少数量パネルの組み合わせ最適化
- 切断計画の自動生成と可視化

### 1.3 期待効果
- 材料ロス削減（目標：効率向上10-20%）
- 計画作業時間短縮（手動→自動）
- 切断ミス削減
- 標準化された切断手順

## 2. 技術要件

### 2.1 基盤技術
- **フレームワーク**: Streamlit
- **言語**: Python 3.9+
- **アルゴリズム**: ギロチンカット制約付き2D Bin Packing
- **可視化**: Plotly/Matplotlib
- **データ処理**: Pandas, NumPy

### 2.2 推奨ライブラリ
```
streamlit >= 1.28.0
pandas >= 2.0.0
numpy >= 1.24.0
plotly >= 5.15.0
matplotlib >= 3.7.0
openpyxl >= 3.1.0  # Excelファイル対応
pydantic >= 2.0.0  # データ検証
aiofiles >= 23.0.0  # 非同期I/O
fastapi >= 0.100.0  # REST API
redis >= 4.6.0  # キャッシュ管理
python-multipart >= 0.0.6  # ファイルアップロード
```

### 2.3 動作環境
- **OS**: Windows 10/11（製造現場想定）
- **メモリ**: 最小4GB、推奨8GB+
- **ブラウザ**: Chrome/Edge（最新版）

## 3. 機能要件

### 3.1 コア機能

#### 3.1.1 データ入力機能
- **パネル情報入力**
  - ID/名称、幅、高さ、数量、材質、板厚
  - テキストベースデータからのパース機能
  - リストデータ形式での一括処理
  - 手動入力フォーム

- **母材情報入力**
  - 幅、高さ、板厚、材質、単価
  - 標準サイズ: 1500mm×3100mm
  - 複数材質対応（鋼板種類別ブロック化）

- **切断順序設定**
  - 鋼板種類によるデータブロック化
  - ブロック内切断順序: 上から下へ
  - 材質別優先度設定

#### 3.1.2 最適化エンジン
- **アルゴリズム選択**
  - First Fit Decreasing (高速)
  - Best Fit Decreasing (高効率)
  - Bottom-Left Fill with Guillotine
  - 遺伝的アルゴリズム（オプション）

- **制約条件**
  - ギロチンカット制約（直線切断のみ）
  - 切断代（ケルフ）考慮
  - パネル回転可/不可設定
  - 最小切断サイズ制限

#### 3.1.3 結果出力機能
- **切断図生成**
  - 2D可視化（母材ごと）
  - パネル配置座標
  - 切断線表示
  - 寸法注記
  - **作業指示図**: 切断順序番号付き詳細図

- **レポート生成**
  - 材料効率率
  - 母材使用枚数
  - 総コスト計算
  - **切断作業指示書**: 
    - 鋼板種類別ブロック
    - 上から下への切断順序
    - 各カットの寸法・位置
  - Excel/PDF出力

- **作業指示レベル出力**
  - ステップバイステップ切断手順
  - 切断線の優先度表示
  - 残材管理情報
  - 品質チェックポイント

### 3.2 UI/UX機能

#### 3.2.1 メイン画面構成
```
┌─────────────────────────────────────────────┐
│ Header: タイトル + 設定メニュー                      │
├─────────────────────────────────────────────┤
│ Left Panel:        │ Right Panel:              │
│ - 母材設定          │ - 結果表示                │
│ - パネル入力        │ - 切断図                  │
│ - 最適化設定        │ - 効率指標                │
│ - 実行ボタン        │ - ダウンロード             │
└─────────────────────────────────────────────┘
```

#### 3.2.2 インタラクティブ要素
- リアルタイム効率計算
- パネル配置のドラッグ&ドロップ編集
- 切断順序のシミュレーション
- パラメータ調整による最適化再実行

### 3.3 データ管理機能
- セッション状態保存
- 計画データのJSON形式保存/読み込み
- 履歴管理（最近の計画）
- テンプレート保存機能

### 3.4 統合・連携機能

#### 3.4.1 外部システム統合
- **ERP/MES連携**
  - REST API経由でのデータ交換
  - CSV/XML形式でのバッチインポート/エクスポート
  - リアルタイム在庫連携
  - 生産計画との同期

#### 3.4.2 エラーハンドリングとリカバリ
```python
class OptimizationError(Exception):
    """最適化エラーの基底クラス"""
    def __init__(self, message: str, recovery_actions: List[str] = None):
        self.message = message
        self.recovery_actions = recovery_actions or []
        super().__init__(self.message)

class TimeoutError(OptimizationError):
    """タイムアウトエラー"""
    def __init__(self, best_solution_so_far: PlacementResult = None):
        super().__init__("最適化がタイムアウトしました")
        self.best_solution = best_solution_so_far
        self.recovery_actions = [
            "現在の最良解を使用",
            "時間制約を緩和して再実行",
            "より高速なアルゴリズムに切り替え"
        ]
```

#### 3.4.3 同時実行制御
- **排他制御**: 複数ユーザーの同時アクセス管理
- **セッション分離**: ユーザーごとの独立した作業環境
- **リソース管理**: CPU/メモリ使用量の監視と制限
- **キュー管理**: 長時間処理のバックグラウンド実行

## 4. データ構造設計

### 4.1 パネルデータ構造
```python
@dataclass
class Panel:
    id: str              # パネルID
    width: float         # 幅 (mm) - 最大1500mm
    height: float        # 高さ (mm) - 最大3100mm
    quantity: int        # 必要数量
    material: str        # 材質（ブロック化キー）
    thickness: float     # 板厚 (mm)
    priority: int        # 優先度 (1-10)
    allow_rotation: bool # 回転許可
    block_order: int     # ブロック内順序（上から下）
    
    def validate_size(self) -> bool:
        """サイズ制約チェック"""
        return (50 <= self.width <= 1500 and 
                50 <= self.height <= 3100)

@dataclass
class RobustTextParser:
    """堅牢なテキストデータ解析クラス"""
    raw_data: str        # 元テキストデータ
    delimiter: str       # 区切り文字
    format_type: str     # データフォーマット種類
    encoding: str = 'utf-8'  # 文字エンコーディング
    unit_system: str = 'mm'  # 単位系（mm/inch）

    def detect_format(self) -> str:
        """データフォーマットを自動検出"""
        if '\t' in self.raw_data[:100]:
            return 'tsv'
        elif ',' in self.raw_data[:100]:
            return 'csv'
        elif self.raw_data.startswith('{'):
            return 'json'
        else:
            return 'fixed_width'

    def validate_and_fix(self, panels: List[Panel]) -> List[Panel]:
        """データ検証と自動修正"""
        fixed_panels = []
        for panel in panels:
            # サイズ制約チェックと修正
            if panel.width > 1500 or panel.height > 3100:
                # 回転を試みる
                if panel.height <= 1500 and panel.width <= 3100:
                    panel.width, panel.height = panel.height, panel.width
            # 単位変換（inch→mm）
            if self.unit_system == 'inch':
                panel.width *= 25.4
                panel.height *= 25.4
            fixed_panels.append(panel)
        return fixed_panels

    def parse_to_panels(self) -> Tuple[List[Panel], List[str]]:
        """テキストデータをパネルリストに変換（エラーリスト付き）"""
        panels = []
        errors = []
        try:
            format_type = self.detect_format() if not self.format_type else self.format_type
            # フォーマット別パース処理
            if format_type == 'csv':
                panels = self._parse_csv()
            elif format_type == 'tsv':
                panels = self._parse_tsv()
            elif format_type == 'json':
                panels = self._parse_json()
            else:
                panels = self._parse_fixed_width()

            # 検証と修正
            panels = self.validate_and_fix(panels)
        except Exception as e:
            errors.append(f"Parse error: {str(e)}")

        return panels, errors
```

### 4.2 母材データ構造
```python
@dataclass
class SteelSheet:
    width: float         # 幅 (mm)
    height: float        # 高さ (mm)
    thickness: float     # 板厚 (mm)
    material: str        # 材質
    cost_per_sheet: float # 単価 (円)
    availability: int    # 在庫数
    priority: int        # 使用優先度
```

### 4.3 配置結果データ構造
```python
@dataclass
class PlacementResult:
    sheet_id: int        # 母材番号
    material_block: str  # 材質ブロック
    panels: List[PlacedPanel]  # 配置パネルリスト
    efficiency: float    # 材料効率 (0-1)
    waste_area: float    # 無駄面積 (mm²)
    cut_length: float    # 総切断長 (mm)
    cost: float         # 材料コスト (円)
    
@dataclass
class CuttingInstruction:
    """切断作業指示"""
    step_number: int     # 作業ステップ番号
    cut_type: str       # 切断種別 ('horizontal'/'vertical')
    start_point: Tuple[float, float]  # 切断開始点
    end_point: Tuple[float, float]    # 切断終了点
    dimension: float    # 切断寸法
    description: str    # 作業説明
    remaining_pieces: List[str]  # 残材情報

@dataclass
class WorkInstruction:
    """作業指示書全体"""
    sheet_id: int
    material_type: str
    total_steps: int
    cutting_sequence: List[CuttingInstruction]
    quality_checkpoints: List[str]
    safety_notes: List[str]
    machine_constraints: Dict[str, Any]  # 機械固有の制約

@dataclass
class QualityCheckpoint:
    """品質チェックポイント"""
    checkpoint_id: str
    step_number: int
    measurement_points: List[Tuple[float, float]]
    tolerance: float  # 許容誤差（mm）
    inspection_method: str  # 検査方法
    critical: bool  # クリティカルチェックポイントか

@dataclass
class MachineConstraints:
    """機械制約"""
    machine_id: str
    max_cutting_length: float  # 最大切断長（mm）
    min_cutting_length: float  # 最小切断長（mm）
    blade_width: float  # ブレード幅（mm）
    cutting_speed: float  # 切断速度（mm/s）
    blade_wear_factor: float  # ブレード摩耗係数
```

## 5. アルゴリズム仕様

### 5.1 基本戦略
1. **前処理**: パネルサイズ・数量による並び替え
2. **配置**: ギロチン制約下での最適位置探索
3. **評価**: 複数解の比較・選択
4. **後処理**: 切断順序最適化

### 5.2 最適化アルゴリズム

#### 5.2.1 階層型最適化戦略
```python
@dataclass
class OptimizationStrategy:
    """問題複雑度に応じた適応的アルゴリズム選択"""
    def select_algorithm(self, complexity: float, time_budget: float) -> str:
        if complexity < 0.3 and time_budget > 1.0:
            return "FFD"  # 高速、約70-75%効率
        elif complexity < 0.7 and time_budget > 5.0:
            return "BFD"  # バランス型、約80-85%効率
        elif time_budget > 30.0:
            return "HYBRID"  # 最適化重視、85%+効率
        else:
            return "FFD_WITH_TIMEOUT"  # タイムアウト付き高速処理

    def estimate_complexity(self, panels: List[Panel]) -> float:
        """パネル数、サイズ多様性、回転可否から複雑度を算出"""
        diversity = len(set((p.width, p.height) for p in panels))
        return min(1.0, (len(panels) * diversity) / 1000)
```

#### 5.2.2 First Fit Decreasing (FFD)
- **特徴**: 高速、実装容易、材料効率70-75%
- **処理時間**: O(n log n)
- **適用条件**: パネル数≤10または時間制約≤1秒
- **手順**:
  1. パネルを面積降順ソート
  2. Bottom-Left戦略で配置
  3. フィットしない場合は新しい母材

#### 5.2.3 Best Fit Decreasing (BFD)
- **特徴**: FFDより高効率、材料効率80-85%
- **処理時間**: O(n² log n)
- **適用条件**: パネル数≤30かつ時間制約≤5秒
- **手順**:
  1. 各パネルに対し全空き領域をスコア計算
  2. 最小無駄面積の位置に配置
  3. ギロチン制約でエリア分割

#### 5.2.4 Hybrid Progressive Optimization
- **特徴**: 段階的最適化、材料効率85%以上
- **処理時間**: 時間制約内でベストエフォート
- **手順**:
  1. FFDで初期解生成（1秒以内）
  2. BFDで改善（5秒まで）
  3. 局所探索で微調整（残り時間）
  4. タイムアウト時は現在の最良解を返却

#### 5.2.5 Bottom-Left-Fill (BLF)
- **特徴**: ギロチン制約に最適化
- **手順**:
  1. 左下詰めでパネル配置
  2. 配置後にギロチン線で領域分割
  3. 重複領域の削除

### 5.3 制約処理

#### 5.3.1 ギロチンカット制約
- 全切断線は母材端から端まで貫通
- 直角切断のみ（L字切断禁止）
- 切断順序の最適化

#### 5.3.2 実用制約
- **サイズ制限**:
  - 最小パネルサイズ: 50mm×50mm
  - 最大パネルサイズ: 3100mm×3100mm
  - 幅（W）制限: 最大1500mm
  - 高さ（H）制限: 最大3100mm
- **切断代（ケルフ）**: 通常3-5mm
- **母材制約**: 1500mm×3100mm以内

## 6. パフォーマンス要件

### 6.1 処理時間目標（現実的な期待値）

#### 複雑度別処理時間
| パネル数 | 複雑度 | アルゴリズム | 目標時間 | 期待効率 |
|---------|-------|------------|---------|----------|
| ≤10 | 低 | FFD | <1秒 | 70-75% |
| ≤10 | 高 | BFD | <3秒 | 80-85% |
| ≤30 | 低 | FFD | <2秒 | 70-75% |
| ≤30 | 高 | BFD/Hybrid | <10秒 | 80-85% |
| ≤50 | 任意 | Hybrid | <15秒 | 75-85% |
| ≤100 | 任意 | Hybrid+Timeout | <30秒 | ベストエフォート |

**注意**: 実際の処理時間はパネルの多様性、回転可否、制約条件により変動

### 6.2 メモリ使用量
- **最大使用量**: 512MB以内（大規模データ時は1GBまで許容）
- **起動時間**: 3秒以内
- **レスポンス時間**: UI操作後1秒以内
- **キャッシュ戦略**: LRUキャッシュで頻繁な計算を最適化

### 6.4 アーキテクチャ最適化
```python
from functools import lru_cache
import asyncio
from concurrent.futures import ProcessPoolExecutor

class PerformanceOptimizer:
    def __init__(self):
        self.cache_size = 1000
        self.executor = ProcessPoolExecutor(max_workers=4)

    @lru_cache(maxsize=1000)
    def calculate_placement_score(self, layout_hash: str) -> float:
        """配置スコアのキャッシュ"""
        pass

    async def optimize_with_progress(self, panels, callback):
        """非同期最適化とプログレス通知"""
        quick_solution = await self.run_ffd_async(panels)
        await callback(quick_solution, progress=0.3)

        if time_available():
            better_solution = await self.run_bfd_async(panels)
            await callback(better_solution, progress=0.7)

        return await self.finalize_solution()
```

### 6.3 最適化品質
- **効率目標**: 手動作業比10-20%向上
- **実行成功率**: 99%以上
- **解の再現性**: 同条件で同一結果

## 7. UI/UX設計仕様

### 7.1 レスポンシブ対応
- **デスクトップ**: 1920×1080以上推奨
- **タブレット**: 1024×768以上対応
- **スマートフォン**: 基本機能のみ

### 7.2 カラーパレット
- **プライマリ**: #1f77b4 (青系)
- **セカンダリ**: #ff7f0e (オレンジ系)  
- **アクセント**: #2ca02c (緑系)
- **警告**: #d62728 (赤系)

### 7.3 操作性
- **直感的操作**: ドラッグ&ドロップサポート
- **視覚的フィードバック**: プログレスバー、ローディング表示
- **エラーハンドリング**: わかりやすいエラーメッセージ

## 8. ファイル構成

```
steel_cutting_optimizer/
├── app.py                 # メインアプリケーション
├── requirements.txt       # 依存関係
├── config/
│   ├── settings.py       # 設定定数
│   ├── algorithms.py     # アルゴリズム設定
│   ├── constraints.py    # 制約条件設定
│   └── performance.py    # 性能プロファイル設定
├── core/
│   ├── models.py         # データ構造定義
│   ├── optimizer.py      # 最適化エンジン
│   ├── guillotine.py     # ギロチンパッキング
│   ├── text_parser.py    # 堅牢なテキストパーサー
│   ├── error_handler.py  # エラーハンドリング
│   ├── cache.py          # キャッシュ管理
│   └── utils.py          # ユーティリティ
├── cutting/
│   ├── instruction.py    # 切断作業指示生成
│   ├── sequence.py       # 切断順序最適化
│   ├── validator.py      # サイズ・制約バリデーター
│   └── quality.py        # 品質チェックポイント管理
├── integration/
│   ├── api.py            # REST APIエンドポイント
│   ├── erp_connector.py  # ERP/MES連携
│   └── session.py        # セッション管理
├── ui/
│   ├── components.py     # UIコンポーネント
│   ├── visualizer.py     # 可視化機能
│   ├── work_instruction_ui.py  # 作業指示UI
│   └── forms.py          # 入力フォーム
├── data/
│   ├── templates/        # テンプレートファイル
│   ├── samples/          # サンプルデータ
│   └── text_formats/     # テキスト形式サンプル
└── tests/
    ├── test_optimizer.py # テストコード
    ├── test_models.py    # モデルテスト
    ├── test_parser.py    # パーサーテスト
    ├── test_constraints.py # 制約テスト
    ├── test_performance.py # 性能テスト
    └── benchmark/        # ベンチマークデータ
```

## 9. 開発段階（現実的なスケジュール）

### Phase 0: 事前準備・要件明確化 (1週間)
- [ ] 要件詳細明確化セッション
- [ ] アルゴリズム性能ベンチマーク
- [ ] 現実的なテストデータセット作成
- [ ] 統合ポイントの特定
- [ ] 性能目標の再設定

### Phase 1: コア機能開発 (3週間)
- [ ] データ構造実装とユニットテスト
- [ ] 堅牢なテキストパーサー実装（複数フォーマット対応）
- [ ] FFDアルゴリズムと性能測定
- [ ] 基本UI構築とプログレス表示
- [ ] サイズ制約検証とエラーハンドリング
- [ ] 簡単な可視化機能（キャッシュ付き）

### Phase 2: 最適化強化 (3週間)
- [ ] BFDアルゴリズム追加
- [ ] ハイブリッドアルゴリズム実装
- [ ] タイムアウト処理と最良解保存
- [ ] ギロチン制約強化
- [ ] 材質別ブロック化処理
- [ ] パフォーマンス最適化（キャッシング、並列処理）

### Phase 3: 作業指示・統合機能 (3週間)
- [ ] 切断作業指示生成機能
- [ ] 品質チェックポイント統合
- [ ] 機械制約考慮
- [ ] ERP/MESシステム統合API
- [ ] Excel/PDF詳細出力
- [ ] 同時アクセス制御

### Phase 4: テスト・最終調整 (2週間)
- [ ] 統合テストとストレステスト
- [ ] 実データでの性能検証
- [ ] 現場ユーザーテスト
- [ ] フィードバック反映
- [ ] パフォーマンスチューニング
- [ ] ドキュメント整備

**合計開発期間**: 12週間（約3ヶ月）

## 10. 成功基準

### 10.1 定量的指標（改訂版）
- 材料効率:
  - FFD: 70-75%以上
  - BFD: 80-85%以上
  - Hybrid: 85%以上（ベストエフォート）
- 処理時間: 複雑度別目標内完了90%以上
- システム稼働率: 99%以上
- サイズ制約: 50-3100mm範囲での100%対応
- エラーリカバリ: タイムアウト時の最良解提供100%

### 10.2 定性的指標
- ユーザビリティ: 直感的操作可能
- 保守性: コード品質・ドキュメント
- 拡張性: 新アルゴリズム追加容易
- **作業実用性**: 
  - 作業指示書の現場での使用可能性
  - テキストデータからの正確な変換
  - 材質別ブロック化の有効性

## 11. 参考文献・アルゴリズム

### 11.1 主要論文
- "The Two-Dimensional Bin Packing Problem" (Lodi et al.)
- "Algorithms for Two-Dimensional Bin Packing" (Berkey & Wang)
- "Guillotine Constraints in Two-dimensional Packing" (Pisinger & Sigurd)

### 11.2 実装参考
- 2D Bin Packing libraries (Python)
- OR-Tools (Google Optimization Tools)
- COIN-OR Cut Stock Problem solvers

## 12. リスク管理と緩和策

### 12.1 技術リスク
| リスク | 影響 | 確率 | 緩和策 |
|--------|------|------|--------|
| アルゴリズム性能不足 | 高 | 中 | 階層型最適化、タイムアウト処理 |
| パーサーエラー | 中 | 高 | 複数フォーマット対応、自動検出 |
| スケーラビリティ | 中 | 中 | キャッシュ、並列処理 |
| 統合の複雑さ | 中 | 中 | API標準化、段階的統合 |

### 12.2 パフォーマンス監視
```python
class PerformanceMonitor:
    """パフォーマンス監視と早期警告"""
    def monitor_optimization(self, process):
        if process.memory_usage > MEMORY_THRESHOLD:
            self.trigger_memory_optimization()
        if process.time_elapsed > TIME_BUDGET * 0.8:
            self.prepare_timeout_recovery()
        if process.efficiency < TARGET_EFFICIENCY:
            self.suggest_algorithm_switch()
```

---

**重要事項**: この仕様書は開発プロセスで継続的に更新・改善していく生きた文書とする。特にアルゴリズム性能の実測値に基づいて目標を調整する。