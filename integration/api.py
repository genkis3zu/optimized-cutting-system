"""
REST API Endpoints for Steel Cutting System Integration
鋼板切断システム統合用REST APIエンドポイント

Provides REST API for external system integration
外部システム統合用のREST APIを提供
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

try:
    from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. API endpoints will be disabled.")

from core.models import Panel, SteelSheet, OptimizationConstraints, PlacementResult
from core.optimizer import create_optimization_engine
from core.algorithms.ffd import create_ffd_algorithm
from core.algorithms.bfd import create_bfd_algorithm
from core.algorithms.hybrid import create_hybrid_algorithm
from cutting.instruction import create_work_instruction_generator
from cutting.quality import create_quality_manager, QualityLevel
from cutting.validator import create_enhanced_validator, ValidationLevel
from cutting.export import create_document_exporter


# API Models
class PanelRequest(BaseModel):
    """Panel request model"""
    id: str
    width: float = Field(..., gt=0, description="Width in mm")
    height: float = Field(..., gt=0, description="Height in mm")
    quantity: int = Field(..., gt=0, description="Quantity")
    material: str = Field(..., description="Material type")
    thickness: float = Field(..., gt=0, description="Thickness in mm")
    priority: int = Field(default=1, ge=1, le=10, description="Priority 1-10")
    allow_rotation: bool = Field(default=True, description="Allow rotation")


class SteelSheetRequest(BaseModel):
    """Steel sheet request model"""
    width: float = Field(default=1500.0, gt=0, description="Width in mm")
    height: float = Field(default=3100.0, gt=0, description="Height in mm")
    thickness: float = Field(default=6.0, gt=0, description="Thickness in mm")
    material: str = Field(default="SS400", description="Material type")
    cost_per_sheet: float = Field(default=10000.0, ge=0, description="Cost per sheet")


class OptimizationRequest(BaseModel):
    """Optimization request model"""
    panels: List[PanelRequest]
    steel_sheet: Optional[SteelSheetRequest] = None
    constraints: Optional[Dict[str, Any]] = None
    algorithm_hint: Optional[str] = Field(default=None, description="Algorithm preference: FFD, BFD, HYBRID")
    include_work_instruction: bool = Field(default=True, description="Generate work instruction")
    include_quality_plan: bool = Field(default=True, description="Generate quality plan")
    validation_level: str = Field(default="standard", description="Validation level: basic, standard, strict, production")


class OptimizationResponse(BaseModel):
    """Optimization response model"""
    optimization_id: str
    status: str
    placement_results: List[Dict[str, Any]]
    work_instructions: Optional[List[Dict[str, Any]]] = None
    quality_plans: Optional[List[Dict[str, Any]]] = None
    validation_report: Optional[Dict[str, Any]] = None
    processing_time: float
    generated_at: str


class WorkInstructionRequest(BaseModel):
    """Work instruction request model"""
    optimization_id: str
    sheet_id: int
    include_quality_checkpoints: bool = Field(default=True)
    export_format: str = Field(default="pdf", description="Export format: pdf, json")


class QualityRecordRequest(BaseModel):
    """Quality record request model"""
    checkpoint_id: str
    inspector: str
    pass_status: bool
    measured_value: Optional[float] = None
    notes: str = ""


# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Steel Cutting Optimization API",
        description="REST API for steel cutting optimization system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global instances
    optimization_engine = None
    work_instruction_generator = None
    quality_manager = None
    validator = None
    document_exporter = None

    # In-memory storage (use database in production)
    optimization_results = {}
    work_instructions = {}
    quality_records = {}


    @app.on_event("startup")
    async def startup_event():
        """Initialize services on startup"""
        global optimization_engine, work_instruction_generator, quality_manager, validator, document_exporter

        logging.info("Starting Steel Cutting Optimization API...")

        # Initialize optimization engine
        optimization_engine = create_optimization_engine()

        # Register algorithms
        optimization_engine.register_algorithm(create_ffd_algorithm())
        optimization_engine.register_algorithm(create_bfd_algorithm())
        optimization_engine.register_algorithm(create_hybrid_algorithm())

        # Initialize other services
        work_instruction_generator = create_work_instruction_generator()
        quality_manager = create_quality_manager()
        validator = create_enhanced_validator()
        document_exporter = create_document_exporter()

        logging.info("Steel Cutting Optimization API started successfully")


    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint with system information"""
        return {
            "name": "Steel Cutting Optimization API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "optimize": "/api/v1/optimize",
                "work_instructions": "/api/v1/work_instructions",
                "quality": "/api/v1/quality",
                "validation": "/api/v1/validation",
                "export": "/api/v1/export"
            }
        }


    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "optimization_engine": optimization_engine is not None,
                "work_instruction_generator": work_instruction_generator is not None,
                "quality_manager": quality_manager is not None,
                "validator": validator is not None,
                "document_exporter": document_exporter is not None
            }
        }


    @app.post("/api/v1/optimize", response_model=OptimizationResponse, tags=["Optimization"])
    async def optimize_cutting(request: OptimizationRequest, background_tasks: BackgroundTasks):
        """
        Optimize cutting layout for given panels
        指定パネルの切断レイアウトを最適化
        """
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        try:
            # Convert request models to domain models
            panels = [
                Panel(
                    id=p.id,
                    width=p.width,
                    height=p.height,
                    quantity=p.quantity,
                    material=p.material,
                    thickness=p.thickness,
                    priority=p.priority,
                    allow_rotation=p.allow_rotation
                )
                for p in request.panels
            ]

            # Steel sheet
            sheet_req = request.steel_sheet or SteelSheetRequest()
            steel_sheet = SteelSheet(
                width=sheet_req.width,
                height=sheet_req.height,
                thickness=sheet_req.thickness,
                material=sheet_req.material,
                cost_per_sheet=sheet_req.cost_per_sheet
            )

            # Constraints
            constraints_dict = request.constraints or {}
            constraints = OptimizationConstraints(**constraints_dict)

            # Validation
            validation_level = ValidationLevel(request.validation_level)
            validator_instance = create_enhanced_validator(validation_level)

            panel_validation = validator_instance.validate_panels(panels, constraints)
            if panel_validation.has_critical_issues:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Panel validation failed",
                        "validation_report": {
                            "overall_result": panel_validation.overall_result.value,
                            "issues": [
                                {
                                    "level": issue.level.value,
                                    "category": issue.category,
                                    "message": issue.message,
                                    "japanese_message": issue.japanese_message
                                }
                                for issue in panel_validation.issues
                            ]
                        }
                    }
                )

            # Run optimization
            start_time = datetime.now()

            placement_results = optimization_engine.optimize(
                panels=panels,
                constraints=constraints,
                algorithm_hint=request.algorithm_hint
            )

            processing_time = (datetime.now() - start_time).total_seconds()

            # Generate work instructions if requested
            work_instructions_data = None
            if request.include_work_instruction and placement_results:
                work_instructions_data = []
                for result in placement_results:
                    work_instruction = work_instruction_generator.generate_work_instruction(
                        result, steel_sheet
                    )
                    work_instructions[f"{optimization_id}_{result.sheet_id}"] = work_instruction
                    work_instructions_data.append(_serialize_work_instruction(work_instruction))

            # Generate quality plans if requested
            quality_plans_data = None
            if request.include_quality_plan and work_instructions_data:
                quality_plans_data = []
                for wi_key in work_instructions:
                    if wi_key.startswith(optimization_id):
                        wi = work_instructions[wi_key]
                        checkpoints = quality_manager.generate_quality_plan(wi)
                        quality_plans_data.append({
                            "sheet_id": wi.sheet_id,
                            "checkpoints": [_serialize_checkpoint(cp) for cp in checkpoints]
                        })

            # Store results
            optimization_results[optimization_id] = {
                "request": request.dict(),
                "placement_results": placement_results,
                "processing_time": processing_time,
                "generated_at": start_time.isoformat()
            }

            return OptimizationResponse(
                optimization_id=optimization_id,
                status="completed",
                placement_results=[_serialize_placement_result(pr) for pr in placement_results],
                work_instructions=work_instructions_data,
                quality_plans=quality_plans_data,
                validation_report={
                    "overall_result": panel_validation.overall_result.value,
                    "pass_rate": panel_validation.pass_rate,
                    "issues_count": len(panel_validation.issues)
                },
                processing_time=processing_time,
                generated_at=start_time.isoformat()
            )

        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization failed: {str(e)}"
            )


    @app.get("/api/v1/optimization/{optimization_id}", tags=["Optimization"])
    async def get_optimization_result(optimization_id: str):
        """Get optimization result by ID"""
        if optimization_id not in optimization_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Optimization result not found"
            )

        result = optimization_results[optimization_id]
        return {
            "optimization_id": optimization_id,
            "status": "completed",
            "placement_results": [_serialize_placement_result(pr) for pr in result["placement_results"]],
            "processing_time": result["processing_time"],
            "generated_at": result["generated_at"]
        }


    @app.post("/api/v1/work_instructions/export", tags=["Work Instructions"])
    async def export_work_instruction(request: WorkInstructionRequest):
        """
        Export work instruction in specified format
        指定形式で作業指示をエクスポート
        """
        wi_key = f"{request.optimization_id}_{request.sheet_id}"

        if wi_key not in work_instructions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Work instruction not found"
            )

        work_instruction = work_instructions[wi_key]

        try:
            if request.export_format.lower() == "pdf":
                # Generate quality checkpoints if requested
                quality_checkpoints = None
                if request.include_quality_checkpoints:
                    quality_checkpoints = quality_manager.generate_quality_plan(work_instruction)

                # Export PDF
                pdf_path = document_exporter.export_work_instruction_pdf(
                    work_instruction,
                    quality_checkpoints
                )

                return FileResponse(
                    pdf_path,
                    media_type="application/pdf",
                    filename=f"work_instruction_{request.sheet_id}.pdf"
                )

            elif request.export_format.lower() == "json":
                return JSONResponse({
                    "work_instruction": _serialize_work_instruction(work_instruction),
                    "export_format": "json",
                    "generated_at": datetime.now().isoformat()
                })

            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported export format. Use 'pdf' or 'json'"
                )

        except Exception as e:
            logging.error(f"Export failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Export failed: {str(e)}"
            )


    @app.post("/api/v1/quality/record", tags=["Quality"])
    async def record_quality_inspection(record: QualityRecordRequest):
        """
        Record quality inspection result
        品質検査結果を記録
        """
        try:
            quality_record = quality_manager.record_inspection_result(
                checkpoint_id=record.checkpoint_id,
                inspector=record.inspector,
                pass_status=record.pass_status,
                measured_value=record.measured_value,
                notes=record.notes
            )

            # Store record
            record_id = f"{record.checkpoint_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            quality_records[record_id] = quality_record

            return {
                "record_id": record_id,
                "status": "recorded",
                "timestamp": quality_record.timestamp.isoformat()
            }

        except Exception as e:
            logging.error(f"Quality record failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Quality record failed: {str(e)}"
            )


    @app.get("/api/v1/quality/report/{optimization_id}", tags=["Quality"])
    async def get_quality_report(optimization_id: str):
        """
        Get quality report for optimization
        最適化の品質レポートを取得
        """
        # Find work instructions for this optimization
        related_wis = [wi for key, wi in work_instructions.items() if key.startswith(optimization_id)]

        if not related_wis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No work instructions found for optimization"
            )

        reports = []
        for wi in related_wis:
            checkpoints = quality_manager.generate_quality_plan(wi)
            report = quality_manager.generate_quality_report(wi, checkpoints)
            reports.append(report)

        return {
            "optimization_id": optimization_id,
            "quality_reports": reports,
            "generated_at": datetime.now().isoformat()
        }


    @app.post("/api/v1/validation/panels", tags=["Validation"])
    async def validate_panels(panels: List[PanelRequest], validation_level: str = "standard"):
        """
        Validate panel specifications
        パネル仕様をバリデーション
        """
        try:
            # Convert to domain models
            domain_panels = [
                Panel(
                    id=p.id,
                    width=p.width,
                    height=p.height,
                    quantity=p.quantity,
                    material=p.material,
                    thickness=p.thickness,
                    priority=p.priority,
                    allow_rotation=p.allow_rotation
                )
                for p in panels
            ]

            # Validate
            validation_level_enum = ValidationLevel(validation_level)
            validator_instance = create_enhanced_validator(validation_level_enum)

            validation_report = validator_instance.validate_panels(domain_panels)

            return {
                "validation_result": validation_report.overall_result.value,
                "pass_rate": validation_report.pass_rate,
                "passed_checks": validation_report.passed_checks,
                "total_checks": validation_report.total_checks,
                "execution_time": validation_report.execution_time,
                "issues": [
                    {
                        "level": issue.level.value,
                        "category": issue.category,
                        "message": issue.message,
                        "japanese_message": issue.japanese_message,
                        "suggestion": issue.suggestion,
                        "affected_items": issue.affected_items
                    }
                    for issue in validation_report.issues
                ]
            }

        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Validation failed: {str(e)}"
            )


    @app.get("/api/v1/algorithms", tags=["System"])
    async def get_available_algorithms():
        """Get list of available optimization algorithms"""
        return {
            "algorithms": [
                {
                    "name": "FFD",
                    "description": "First Fit Decreasing - Fast optimization",
                    "target_efficiency": "70-75%",
                    "performance": "<1 second for ≤10 panels"
                },
                {
                    "name": "BFD",
                    "description": "Best Fit Decreasing - Better efficiency",
                    "target_efficiency": "80-85%",
                    "performance": "<5 seconds for ≤30 panels"
                },
                {
                    "name": "HYBRID",
                    "description": "Hybrid multi-strategy optimization",
                    "target_efficiency": "85%+",
                    "performance": "<30 seconds for complex problems"
                }
            ]
        }


    @app.get("/api/v1/materials", tags=["System"])
    async def get_supported_materials():
        """Get list of supported materials"""
        return {
            "materials": [
                {
                    "code": "SS400",
                    "name": "Carbon Steel",
                    "japanese_name": "一般構造用圧延鋼材",
                    "applications": ["structural", "general"]
                },
                {
                    "code": "SUS304",
                    "name": "Stainless Steel 304",
                    "japanese_name": "ステンレス鋼304",
                    "applications": ["food_grade", "chemical"]
                },
                {
                    "code": "SUS316",
                    "name": "Stainless Steel 316",
                    "japanese_name": "ステンレス鋼316",
                    "applications": ["marine", "chemical"]
                },
                {
                    "code": "AL6061",
                    "name": "Aluminum 6061",
                    "japanese_name": "アルミニウム合金6061",
                    "applications": ["lightweight", "aerospace"]
                },
                {
                    "code": "AL5052",
                    "name": "Aluminum 5052",
                    "japanese_name": "アルミニウム合金5052",
                    "applications": ["marine", "general"]
                }
            ]
        }


    # Helper functions
    def _serialize_placement_result(placement_result: PlacementResult) -> Dict[str, Any]:
        """Serialize placement result for API response"""
        return {
            "sheet_id": placement_result.sheet_id,
            "material_block": placement_result.material_block,
            "efficiency": placement_result.efficiency,
            "waste_area": placement_result.waste_area,
            "cut_length": placement_result.cut_length,
            "cost": placement_result.cost,
            "algorithm": placement_result.algorithm,
            "processing_time": placement_result.processing_time,
            "timestamp": placement_result.timestamp.isoformat(),
            "panels_placed": len(placement_result.panels),
            "panels": [
                {
                    "panel_id": placed.panel.id,
                    "x": placed.x,
                    "y": placed.y,
                    "width": placed.actual_width,
                    "height": placed.actual_height,
                    "rotated": placed.rotated,
                    "material": placed.panel.material
                }
                for placed in placement_result.panels
            ]
        }


    def _serialize_work_instruction(work_instruction) -> Dict[str, Any]:
        """Serialize work instruction for API response"""
        return {
            "sheet_id": work_instruction.sheet_id,
            "material_type": work_instruction.material_type,
            "total_steps": work_instruction.total_steps,
            "estimated_total_time": work_instruction.estimated_total_time,
            "cutting_sequence": [
                {
                    "step_number": inst.step_number,
                    "cut_type": inst.cut_type.value,
                    "start_point": inst.start_point,
                    "end_point": inst.end_point,
                    "dimension": inst.dimension,
                    "estimated_time": inst.estimated_time
                }
                for inst in work_instruction.cutting_sequence
            ],
            "generated_at": work_instruction.generated_at.isoformat()
        }


    def _serialize_checkpoint(checkpoint) -> Dict[str, Any]:
        """Serialize quality checkpoint for API response"""
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "step_number": checkpoint.step_number,
            "checkpoint_type": checkpoint.checkpoint_type.value,
            "description": checkpoint.description,
            "japanese_description": checkpoint.japanese_description,
            "inspection_method": checkpoint.inspection_method.value,
            "critical_flag": checkpoint.critical_flag,
            "estimated_time": checkpoint.estimated_time
        }


    # Main function to run the API
    def run_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server"""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required to run the API server")

        uvicorn.run(
            "integration.api:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )


else:
    # Fallback when FastAPI is not available
    def run_api(*args, **kwargs):
        raise ImportError("FastAPI is required to run the API server. Install with: pip install fastapi uvicorn")