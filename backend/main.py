"""
FastAPI service for the Financial Chatbot.

Provides a REST API endpoint `/ask` that processes natural language queries
through the complete pipeline and returns structured results.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent  # project root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.orchestrator import PipelineOrchestrator, PipelineResult

# Default database path
DEFAULT_DB_PATH = os.getenv("DUCKDB_PATH", ROOT_DIR / "data/db/trial_balance.duckdb")

# Global orchestrator instance (lazy-loaded)
_orchestrator: Optional[PipelineOrchestrator] = None


def get_orchestrator() -> PipelineOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        db_path = os.getenv("DUCKDB_PATH", DEFAULT_DB_PATH)
        # Ensure database path exists
        if not Path(db_path).exists():
            raise RuntimeError(
                f"Database not found at {db_path}. "
                f"Please ensure the database exists or set DUCKDB_PATH environment variable."
            )
        _orchestrator = PipelineOrchestrator(db_path)
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    yield
    # Shutdown
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.close()
        _orchestrator = None


# Initialize FastAPI app
app = FastAPI(
    title="Financial Chatbot API",
    description="Natural language interface for financial data queries",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str = Field(..., description="Natural language query", min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer_text: str = Field(..., description="Natural language answer")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata and intermediate results")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Financial Chatbot API",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint with database status."""
    try:
        orchestrator = get_orchestrator()
        db_path = orchestrator.database_path
        db_exists = Path(db_path).exists()
        return {
            "status": "ok" if db_exists else "degraded",
            "database_path": db_path,
            "database_exists": db_exists,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "database_path": os.getenv("DUCKDB_PATH", DEFAULT_DB_PATH),
            "database_exists": False,
        }


@app.post("/ask", response_model=QueryResponse)
async def ask_query(request: QueryRequest) -> QueryResponse:
    """
    Process a natural language query and return structured results.
    
    This endpoint:
    1. Parses time expressions
    2. Classifies the query topic
    3. Links entities (accounts)
    4. Generates SQL with PICARD validation
    5. Executes the SQL query
    6. Generates a natural language answer
    
    Returns:
        QueryResponse with answer_text, sql, and metadata containing
        all intermediate pipeline outputs for debugging.
    """
    try:
        # Get orchestrator
        orchestrator = get_orchestrator()
        
        # Process query (run in thread pool to avoid blocking)
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result: PipelineResult = await loop.run_in_executor(
                executor,
                orchestrator.process_query,
                request.query,
            )
        
        # Extract response data
        answer_text = result.answer or "I couldn't generate an answer for your query."
        sql = result.generated_sql
        
        # Build metadata with all intermediate outputs
        metadata: Dict[str, Any] = {
            "runtime_seconds": result.runtime_seconds,
            "query_classification": result.query_classification,
            "time_parse_result": result.time_parse_result,
            "entity_links": result.entity_links,
            "validation_status": result.validation_status,
            "sql_execution_result": {
                "rows": result.sql_execution_result.get("rows") if result.sql_execution_result else None,
                "columns": result.sql_execution_result.get("columns") if result.sql_execution_result else None,
            } if result.sql_execution_result else None,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        
        return QueryResponse(
            answer_text=answer_text,
            sql=sql,
            metadata=metadata,
        )
        
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Enable auto-reload in development
    )
