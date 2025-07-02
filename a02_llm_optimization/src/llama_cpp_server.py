"""
Llama.cpp Server for A02 LLM Optimization
High-performance inference server using llama.cpp with GGUF models
Optimized for 4GB VRAM + Core i5 CPU
"""

import os
import json
import time
import asyncio
import logging
import subprocess
import threading
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import psutil
import GPUtil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp server"""
    model_path: str = "./models/fine_tuned_model.gguf"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # llama.cpp parameters
    n_ctx: int = 2048  # Context length
    n_batch: int = 512  # Batch size for prompt processing
    n_threads: int = -1  # CPU threads (-1 = auto)
    n_gpu_layers: int = 35  # GPU layers (adjust based on VRAM)
    
    # Performance settings
    use_mmap: bool = True
    use_mlock: bool = False
    numa: bool = False
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    
    # Server settings
    timeout: float = 300.0  # 5 minutes
    max_concurrent_requests: int = 4
    
    # Monitoring
    enable_metrics: bool = True
    log_requests: bool = True

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages")
    model: Optional[str] = Field(None, description="Model to use")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    top_k: Optional[int] = Field(40, description="Top-k sampling")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Stream response")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class LlamaCppServer:
    """
    High-performance llama.cpp server for A02 LLM optimization
    
    Features:
    - GGUF model support
    - GPU acceleration with fallback to CPU
    - OpenAI-compatible API
    - Streaming responses
    - Performance monitoring
    - Concurrent request handling
    """
    
    def __init__(self, config: LlamaCppConfig):
        self.config = config
        self.app = FastAPI(
            title="A02 LLM Optimization Server",
            description="High-performance inference server using llama.cpp",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Server state
        self.llama_process = None
        self.is_running = False
        self.request_count = 0
        self.total_tokens = 0
        self.active_requests = 0
        
        # Performance metrics
        self.metrics = {
            "requests_total": 0,
            "requests_active": 0,
            "tokens_generated": 0,
            "average_latency": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": 0.0
        }
        
        # Setup routes
        self._setup_routes()
        
        # Start background monitoring
        if config.enable_metrics:
            self._start_monitoring()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.is_running else "unhealthy",
                "model_loaded": self.llama_process is not None,
                "active_requests": self.active_requests,
                "uptime": time.time() - getattr(self, 'start_time', time.time())
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus-style metrics endpoint"""
            self._update_metrics()
            return self.metrics
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            return await self._handle_chat_completion(request)
        
        @self.app.post("/v1/completions")
        async def completions(request: Dict[str, Any]):
            """OpenAI-compatible completions endpoint"""
            return await self._handle_completion(request)
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models"""
            return {
                "object": "list",
                "data": [
                    {
                        "id": "llama-cpp",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "a02-optimization"
                    }
                ]
            }
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while self.is_running:
                self._update_metrics()
                time.sleep(10)  # Update every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"] = memory.percent
            
            # GPU usage
            if GPUtil.getGPUs():
                gpu = GPUtil.getGPUs()[0]
                self.metrics["gpu_usage"] = gpu.load * 100
            
            # Request metrics
            self.metrics["requests_active"] = self.active_requests
            
        except Exception as e:
            logger.warning(f"Metrics update failed: {e}")
    
    def _check_system_resources(self) -> bool:
        """Check if system has enough resources"""
        try:
            # Check if model file exists
            if not os.path.exists(self.config.model_path):
                logger.error(f"Model file not found: {self.config.model_path}")
                return False
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2:
                logger.warning(f"Low memory: {available_gb:.1f}GB available")
                return False
            
            # Check GPU if available
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_memory_gb = gpu.memoryFree / 1024
                logger.info(f"GPU: {gpu.name}, Free memory: {gpu_memory_gb:.1f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False
    
    def _start_llama_cpp_server(self) -> bool:
        """Start llama.cpp server process"""
        try:
            logger.info("Starting llama.cpp server...")
            
            # Build command
            cmd = [
                "llama-server",  # or "llama-cpp-python[server]"
                "--model", self.config.model_path,
                "--host", self.config.host,
                "--port", str(self.config.port + 1),  # Internal port
                "--ctx-size", str(self.config.n_ctx),
                "--batch-size", str(self.config.n_batch),
                "--threads", str(self.config.n_threads) if self.config.n_threads > 0 else str(os.cpu_count()),
                "--n-gpu-layers", str(self.config.n_gpu_layers),
            ]
            
            # Add optional parameters
            if self.config.use_mmap:
                cmd.append("--mmap")
            if self.config.use_mlock:
                cmd.append("--mlock")
            if self.config.numa:
                cmd.append("--numa")
            
            # Start process
            self.llama_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            if self.llama_process.poll() is None:
                logger.info("llama.cpp server started successfully")
                self.is_running = True
                return True
            else:
                logger.error("llama.cpp server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start llama.cpp server: {e}")
            return False
    
    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt"""
        prompt = ""
        
        for message in messages:
            if message.role == "system":
                prompt += f"System: {message.content}\n"
            elif message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        prompt += "Assistant: "
        return prompt
    
    async def _handle_chat_completion(self, request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
        """Handle chat completion request"""
        start_time = time.time()
        self.active_requests += 1
        self.request_count += 1
        
        try:
            # Check rate limiting
            if self.active_requests > self.config.max_concurrent_requests:
                raise HTTPException(status_code=429, detail="Too many concurrent requests")
            
            # Format prompt
            prompt = self._format_chat_prompt(request.messages)
            
            # Prepare generation parameters
            gen_params = {
                "prompt": prompt,
                "temperature": request.temperature or self.config.temperature,
                "top_p": request.top_p or self.config.top_p,
                "top_k": request.top_k or self.config.top_k,
                "max_tokens": request.max_tokens or self.config.max_tokens,
                "stop": request.stop or ["User:", "System:"],
            }
            
            if request.stream:
                return await self._stream_response(gen_params, start_time)
            else:
                return await self._generate_response(gen_params, start_time)
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.active_requests -= 1
    
    async def _generate_response(self, gen_params: Dict[str, Any], start_time: float) -> ChatCompletionResponse:
        """Generate non-streaming response"""
        try:
            # Call llama.cpp API (placeholder - implement actual API call)
            # This would typically use httpx to call the llama.cpp server
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:{self.config.port + 1}/completion",
                    json=gen_params,
                    timeout=self.config.timeout
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
                
                result = response.json()
                
                # Format OpenAI-compatible response
                completion_response = ChatCompletionResponse(
                    id=f"chatcmpl-{int(time.time())}",
                    created=int(time.time()),
                    model="llama-cpp",
                    choices=[
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": result.get("content", "")
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    usage={
                        "prompt_tokens": result.get("tokens_evaluated", 0),
                        "completion_tokens": result.get("tokens_predicted", 0),
                        "total_tokens": result.get("tokens_evaluated", 0) + result.get("tokens_predicted", 0)
                    }
                )
                
                # Update metrics
                latency = time.time() - start_time
                self.metrics["average_latency"] = (self.metrics["average_latency"] * (self.request_count - 1) + latency) / self.request_count
                self.metrics["requests_total"] += 1
                self.metrics["tokens_generated"] += completion_response.usage["completion_tokens"]
                
                return completion_response
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _stream_response(self, gen_params: Dict[str, Any], start_time: float):
        """Generate streaming response"""
        async def generate():
            try:
                # Implement streaming logic here
                # This is a placeholder - actual implementation would stream from llama.cpp
                
                import httpx
                gen_params["stream"] = True
                
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"http://localhost:{self.config.port + 1}/completion",
                        json=gen_params,
                        timeout=self.config.timeout
                    ) as response:
                        
                        async for chunk in response.aiter_lines():
                            if chunk:
                                # Parse SSE format
                                if chunk.startswith("data: "):
                                    data = chunk[6:]
                                    if data == "[DONE]":
                                        break
                                    
                                    try:
                                        chunk_data = json.loads(data)
                                        
                                        # Format as OpenAI streaming response
                                        streaming_chunk = {
                                            "id": f"chatcmpl-{int(time.time())}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "llama-cpp",
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": chunk_data.get("content", "")
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        
                                        yield f"data: {json.dumps(streaming_chunk)}\n\n"
                                        
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Send final chunk
                        final_chunk = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": "llama-cpp",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        
                        yield f"data: {json.dumps(final_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        
            except Exception as e:
                logger.error(f"Streaming failed: {e}")
                error_chunk = {
                    "error": {
                        "message": str(e),
                        "type": "server_error"
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    async def _handle_completion(self, request: Dict[str, Any]):
        """Handle completion request (non-chat)"""
        # Similar to chat completion but for raw text completion
        # Implementation would be similar to _handle_chat_completion
        pass
    
    def start(self):
        """Start the server"""
        try:
            logger.info("Starting A02 LLM Optimization Server...")
            
            # Check system resources
            if not self._check_system_resources():
                logger.error("Insufficient system resources")
                return False
            
            # Start llama.cpp server
            if not self._start_llama_cpp_server():
                logger.error("Failed to start llama.cpp server")
                return False
            
            self.start_time = time.time()
            
            # Start FastAPI server
            uvicorn.run(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            
        except Exception as e:
            logger.error(f"Server startup failed: {e}")
            return False
    
    def stop(self):
        """Stop the server"""
        try:
            logger.info("Stopping A02 LLM Optimization Server...")
            
            self.is_running = False
            
            if self.llama_process:
                self.llama_process.terminate()
                self.llama_process.wait(timeout=10)
                logger.info("llama.cpp server stopped")
            
        except Exception as e:
            logger.error(f"Server shutdown failed: {e}")

def main():
    """Main entry point"""
    
    # Configuration for 4GB VRAM + Core i5
    config = LlamaCppConfig(
        model_path="./models/fine_tuned_model.gguf",
        host="0.0.0.0",
        port=8000,
        n_ctx=2048,
        n_batch=512,
        n_threads=4,  # Core i5 typically has 4 cores
        n_gpu_layers=20,  # Adjust based on 4GB VRAM
        temperature=0.7,
        max_tokens=512,
        max_concurrent_requests=2,  # Conservative for 4GB VRAM
    )
    
    # Create and start server
    server = LlamaCppServer(config)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        server.stop()

if __name__ == "__main__":
    main()

