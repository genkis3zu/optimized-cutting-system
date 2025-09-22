# API リファレンス / API Reference

鋼板切断最適化システム REST API 仕様書

## 📖 概要 / Overview

本APIは鋼板切断最適化システムの外部連携機能を提供します。ERP/MESシステムからの切断計画作成、最適化実行、作業指示書生成が可能です。

*This API provides external integration capabilities for the steel cutting optimization system, enabling cutting plan creation, optimization execution, and work instruction generation from ERP/MES systems.*

## 🚀 基本情報 / Basic Information

- **ベースURL**: `http://localhost:8000`
- **認証**: 現在は未実装（将来のバージョンで対応予定）
- **コンテンツタイプ**: `application/json`
- **APIバージョン**: v1.0.0

## 📋 エンドポイント一覧 / Endpoints

### 1. 切断最適化 / Optimization

#### `POST /api/v1/optimize`
パネル情報から切断最適化を実行

**リクエスト例 / Request Example:**
```json
{
  "panels": [
    {
      "id": "PANEL_001",
      "width": 300.0,
      "height": 400.0,
      "quantity": 2,
      "material": "SECC",
      "thickness": 1.6,
      "priority": 1,
      "allow_rotation": true
    }
  ],
  "steel_sheet": {
    "width": 1500.0,
    "height": 3100.0,
    "thickness": 1.6,
    "material": "SECC",
    "cost_per_sheet": 15000.0
  },
  "algorithm_hint": "BFD",
  "include_work_instruction": true,
  "include_quality_plan": true,
  "validation_level": "standard"
}
```

**レスポンス例 / Response Example:**
```json
{
  "optimization_id": "opt_20250122_001",
  "status": "completed",
  "placement_results": [
    {
      "sheet_id": 1,
      "material_block": "SECC",
      "efficiency": 0.85,
      "panels": [
        {
          "panel_id": "PANEL_001",
          "x": 0,
          "y": 0,
          "width": 300,
          "height": 400,
          "rotated": false
        }
      ],
      "waste_area": 4335000,
      "cost": 15000.0
    }
  ],
  "processing_time": 2.45,
  "generated_at": "2025-01-22T10:30:00Z"
}
```

### 2. 作業指示書生成 / Work Instruction Generation

#### `POST /api/v1/work-instruction`
最適化結果から作業指示書を生成

**リクエスト例:**
```json
{
  "optimization_id": "opt_20250122_001",
  "sheet_id": 1,
  "include_quality_checkpoints": true,
  "export_format": "pdf"
}
```

**レスポンス例:**
```json
{
  "instruction_id": "inst_20250122_001",
  "sheet_id": 1,
  "total_steps": 15,
  "estimated_time_minutes": 25.5,
  "cutting_sequence": [
    {
      "step_number": 1,
      "cut_type": "horizontal",
      "start_point": [0, 400],
      "end_point": [1500, 400],
      "dimension": 1500,
      "description": "First horizontal cut at 400mm"
    }
  ],
  "quality_checkpoints": [
    {
      "checkpoint_id": "QC_001",
      "step_number": 5,
      "measurement_points": [[150, 200]],
      "tolerance": 0.5,
      "inspection_method": "caliper"
    }
  ],
  "download_url": "/api/v1/download/instruction/inst_20250122_001.pdf"
}
```

### 3. 材料在庫管理 / Material Inventory Management

#### `GET /api/v1/materials`
材料在庫一覧の取得

**レスポンス例:**
```json
{
  "materials": [
    {
      "material_code": "SECC_1.6_1500x3100",
      "material_type": "SECC",
      "thickness": 1.6,
      "width": 1500,
      "height": 3100,
      "area": 4650000,
      "availability": 50,
      "cost_per_sheet": 15000.0,
      "last_updated": "2025-01-22T09:00:00Z"
    }
  ],
  "total_materials": 25,
  "summary": {
    "total_sheets": 1250,
    "total_value": 18750000,
    "material_types": 8
  }
}
```

#### `POST /api/v1/materials`
新しい材料の追加

**リクエスト例:**
```json
{
  "material_code": "SS400_2.0_1500x3100",
  "material_type": "SS400",
  "thickness": 2.0,
  "width": 1500,
  "height": 3100,
  "cost_per_sheet": 12000.0,
  "availability": 100,
  "supplier": "Steel Corp"
}
```

#### `PUT /api/v1/materials/{material_code}`
材料在庫の更新

#### `DELETE /api/v1/materials/{material_code}`
材料の削除

### 4. バリデーション / Validation

#### `POST /api/v1/validate`
パネル情報のバリデーション

**リクエスト例:**
```json
{
  "panels": [
    {
      "id": "PANEL_001",
      "width": 300.0,
      "height": 400.0,
      "material": "SECC",
      "thickness": 1.6
    }
  ],
  "validation_level": "production"
}
```

**レスポンス例:**
```json
{
  "validation_status": "passed",
  "errors": [],
  "warnings": [
    "Panel PANEL_001: Consider rotation for better efficiency"
  ],
  "material_availability": {
    "SECC": {
      "available": true,
      "matching_sheets": 15,
      "sufficient_inventory": true
    }
  },
  "size_constraints": {
    "all_panels_valid": true,
    "oversized_panels": []
  }
}
```

### 5. 品質管理 / Quality Management

#### `POST /api/v1/quality/record`
品質検査結果の記録

**リクエスト例:**
```json
{
  "checkpoint_id": "QC_001",
  "inspector": "田中太郎",
  "pass_status": true,
  "measured_value": 299.8,
  "notes": "寸法精度良好"
}
```

#### `GET /api/v1/quality/report/{optimization_id}`
品質レポートの取得

### 6. ファイルダウンロード / File Downloads

#### `GET /api/v1/download/instruction/{instruction_id}.pdf`
作業指示書PDFのダウンロード

#### `GET /api/v1/download/report/{optimization_id}.xlsx`
最適化レポートExcelのダウンロード

#### `GET /api/v1/download/layout/{optimization_id}.png`
切断レイアウト図のダウンロード

### 7. システム情報 / System Information

#### `GET /api/v1/health`
システムヘルスチェック

**レスポンス例:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "algorithm_status": {
    "FFD": "available",
    "BFD": "available",
    "HYBRID": "available"
  },
  "memory_usage": 245.5,
  "active_optimizations": 2
}
```

#### `GET /api/v1/algorithms`
利用可能なアルゴリズムの一覧

**レスポンス例:**
```json
{
  "algorithms": [
    {
      "name": "FFD",
      "display_name": "First Fit Decreasing",
      "description": "高速処理に適したアルゴリズム",
      "typical_efficiency": "70-75%",
      "recommended_for": "小規模データ（≤20パネル）"
    },
    {
      "name": "BFD",
      "display_name": "Best Fit Decreasing",
      "description": "効率重視のバランス型アルゴリズム",
      "typical_efficiency": "80-85%",
      "recommended_for": "中規模データ（≤50パネル）"
    },
    {
      "name": "HYBRID",
      "display_name": "Hybrid Algorithm",
      "description": "段階的最適化による高効率アルゴリズム",
      "typical_efficiency": "85%+",
      "recommended_for": "大規模データ・高精度要求"
    }
  ]
}
```

## 🔧 リクエスト・レスポンス仕様 / Request/Response Specifications

### データ型定義 / Data Types

#### Panel (パネル)
```typescript
interface Panel {
  id: string;                    // パネルID
  width: number;                 // 幅（mm）
  height: number;                // 高さ（mm）
  quantity: number;              // 数量
  material: string;              // 材質
  thickness: number;             // 板厚（mm）
  priority?: number;             // 優先度（1-10）
  allow_rotation?: boolean;      // 回転許可
}
```

#### SteelSheet (母材)
```typescript
interface SteelSheet {
  width: number;                 // 幅（mm）
  height: number;                // 高さ（mm）
  thickness: number;             // 板厚（mm）
  material: string;              // 材質
  cost_per_sheet: number;        // 単価（円）
}
```

#### PlacementResult (配置結果)
```typescript
interface PlacementResult {
  sheet_id: number;              // シートID
  material_block: string;        // 材質ブロック
  efficiency: number;            // 効率（0-1）
  panels: PlacedPanel[];         // 配置されたパネル
  waste_area: number;            // 無駄面積（mm²）
  cost: number;                  // コスト（円）
}
```

### エラーレスポンス / Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "パネルサイズが制約を超えています",
    "details": {
      "panel_id": "PANEL_001",
      "max_width": 1500,
      "actual_width": 1600
    }
  },
  "timestamp": "2025-01-22T10:30:00Z",
  "request_id": "req_20250122_001"
}
```

### HTTPステータスコード / HTTP Status Codes

- `200 OK` - 成功
- `201 Created` - 作成成功
- `400 Bad Request` - リクエストエラー
- `404 Not Found` - リソースが見つからない
- `422 Unprocessable Entity` - バリデーションエラー
- `500 Internal Server Error` - サーバーエラー

## 🔐 認証・セキュリティ / Authentication & Security

現在のバージョンでは認証機能は未実装ですが、将来のバージョンで以下の機能を予定しています：

- **API キー認証**
- **JWT トークン認証**
- **IP アドレス制限**
- **レート制限**

## 📊 使用例 / Usage Examples

### Python での使用例
```python
import requests
import json

# 最適化リクエスト
api_url = "http://localhost:8000/api/v1/optimize"
data = {
    "panels": [
        {
            "id": "PANEL_001",
            "width": 300.0,
            "height": 400.0,
            "quantity": 1,
            "material": "SECC",
            "thickness": 1.6
        }
    ],
    "algorithm_hint": "BFD"
}

response = requests.post(api_url, json=data)
result = response.json()

print(f"最適化完了: {result['optimization_id']}")
print(f"効率: {result['placement_results'][0]['efficiency']:.1%}")
```

### cURL での使用例
```bash
# 最適化実行
curl -X POST "http://localhost:8000/api/v1/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "panels": [
      {
        "id": "PANEL_001",
        "width": 300.0,
        "height": 400.0,
        "quantity": 1,
        "material": "SECC",
        "thickness": 1.6
      }
    ]
  }'

# 材料在庫確認
curl -X GET "http://localhost:8000/api/v1/materials"

# システム状態確認
curl -X GET "http://localhost:8000/api/v1/health"
```

## 🚀 API サーバーの起動 / Starting API Server

```bash
# 開発用サーバーの起動
uvicorn integration.api:app --reload --port 8000

# 本番用サーバーの起動
uvicorn integration.api:app --host 0.0.0.0 --port 8000 --workers 4

# API仕様書の確認
# ブラウザで http://localhost:8000/docs にアクセス
```

## 📚 関連ドキュメント / Related Documentation

- [📋 プロジェクト仕様書](../steel_cutting_spec.md)
- [🏗️ アーキテクチャ概要](architecture.md)
- [📖 ユーザーガイド](user_guide.md)
- [⚙️ 開発ガイド](../CLAUDE.md)

---

**最終更新**: 2025年1月22日 | **APIバージョン**: v1.0.0