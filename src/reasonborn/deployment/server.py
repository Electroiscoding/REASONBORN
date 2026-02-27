"""
Production FastAPI Inference Server
======================================
REST/SSE API for serving ReasonBorn model inference.
Per ReasonBorn.md Section 5.5.
"""

import os
import time
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any

import torch

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

logger = logging.getLogger("reasonborn.server")


# ── Request / Response Models ──

if HAS_FASTAPI:
    class GenerateRequest(BaseModel):
        prompt: str = Field(..., min_length=1, max_length=8192)
        max_tokens: int = Field(512, ge=1, le=4096)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        top_k: int = Field(50, ge=0)
        top_p: float = Field(0.9, ge=0.0, le=1.0)
        stream: bool = False
        include_proof: bool = False

    class CompletionRequest(BaseModel):
        model: str = "reasonborn-500m"
        prompt: str = Field(..., min_length=1)
        max_tokens: int = Field(512, ge=1, le=4096)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        stream: bool = False

    class GenerateResponse(BaseModel):
        text: str
        confidence: float
        tokens_generated: int
        latency_ms: float
        proof: Optional[Dict[str, Any]] = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        device: str
        uptime_seconds: float


class InferenceServer:
    """
    Production inference server for ReasonBorn.
    Supports batch inference, streaming via SSE, and health monitoring.
    """

    def __init__(self, model: Any = None, device: Optional[str] = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._start_time = time.time()
        self._request_count = 0
        self._total_tokens = 0

    def load_model(self, model: Any) -> None:
        """Load or replace the inference model."""
        self.model = model
        if hasattr(model, 'to'):
            self.model.to(self.device)
        if hasattr(model, 'eval'):
            self.model.eval()
        logger.info(f"Model loaded on {self.device}")

    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 0.7, include_proof: bool = False
                 ) -> Dict[str, Any]:
        """Synchronous generation."""
        if self.model is None:
            raise RuntimeError("No model loaded")

        start = time.time()
        self._request_count += 1

        if hasattr(self.model, 'generate'):
            result = self.model.generate(
                prompt, max_tokens=max_tokens,
                temperature=temperature)
        else:
            result = f"Model does not support generation. Prompt: {prompt}"

        latency = (time.time() - start) * 1000

        # Extract structured result
        if isinstance(result, dict):
            text = result.get('answer', result.get('text', str(result)))
            confidence = result.get('confidence', 0.0)
            proof = result.get('proof') if include_proof else None
        else:
            text = str(result)
            confidence = 0.0
            proof = None

        tokens = len(text.split())
        self._total_tokens += tokens

        return {
            'text': text,
            'confidence': confidence,
            'tokens_generated': tokens,
            'latency_ms': latency,
            'proof': proof,
        }

    def create_app(self) -> Any:
        """Create and configure the FastAPI application."""
        if not HAS_FASTAPI:
            raise ImportError(
                "FastAPI is required. Install with: pip install fastapi uvicorn")

        app = FastAPI(
            title="ReasonBorn Inference Server",
            version="1.0.0",
            description="Production serving for ReasonBorn SS-SLM",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(
                status="ok",
                model_loaded=server.model is not None,
                device=server.device,
                uptime_seconds=time.time() - server._start_time,
            )

        @app.get("/metrics")
        async def metrics():
            return {
                "requests_total": server._request_count,
                "tokens_total": server._total_tokens,
                "uptime_seconds": time.time() - server._start_time,
                "device": server.device,
            }

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(req: GenerateRequest):
            try:
                if req.stream:
                    return StreamingResponse(
                        server._stream_generate(req),
                        media_type="text/event-stream")
                result = server.generate(
                    req.prompt, req.max_tokens, req.temperature,
                    req.include_proof)
                return GenerateResponse(**result)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/v1/completions")
        async def completions(req: CompletionRequest):
            try:
                result = server.generate(
                    req.prompt, req.max_tokens, req.temperature)
                return {
                    "id": f"cmpl-{server._request_count}",
                    "object": "text_completion",
                    "model": req.model,
                    "choices": [{"text": result['text'], "index": 0,
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(req.prompt.split()),
                              "completion_tokens": result['tokens_generated'],
                              "total_tokens": len(req.prompt.split()) + result['tokens_generated']},
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return app

    async def _stream_generate(self, req: Any):
        """SSE streaming generation."""
        result = self.generate(
            req.prompt, req.max_tokens, req.temperature, req.include_proof)
        text = result['text']
        words = text.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i + 3])
            data = json.dumps({"text": chunk, "done": False})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """Start the server."""
        import uvicorn
        app = self.create_app()
        uvicorn.run(app, host=host, port=port)
