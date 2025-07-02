from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import httpx
import time
import os
import redis.asyncio as redis
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Initialize FastAPI app
app = FastAPI(
    title="Optimized AI System API Gateway",
    description="API Gateway for A02 LLM Optimization and A05 Knowledge Base services",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "service", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint", "service"]
)
IN_FLIGHT_REQUESTS = Gauge(
    "http_in_flight_requests", "Number of in-flight requests", ["service"]
)

# Service URLs from environment variables
A02_SERVICE_URL = os.getenv("A02_SERVICE_URL", "http://a02-llm-service:8000")
A05_SERVICE_URL = os.getenv("A05_SERVICE_URL", "http://a05-knowledge-base:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis-cluster:6379")

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "1000")) # requests per window
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))     # seconds

# Redis client for rate limiting
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    print("Connected to Redis for rate limiting.")
    # Start Prometheus HTTP server for metrics
    start_http_server(8000) # Expose metrics on port 8000
    print("Prometheus metrics server started on port 8000.")

@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
        print("Redis connection closed.")

async def _reverse_proxy(request: Request, service_url: str, service_name: str):
    start_time = time.time()
    IN_FLIGHT_REQUESTS.labels(service=service_name).inc()

    # Rate Limiting
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    current_requests = await redis_client.incr(key)
    if current_requests == 1:
        await redis_client.expire(key, RATE_LIMIT_WINDOW)
    
    if current_requests > RATE_LIMIT_REQUESTS:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, service=service_name, status_code=429).inc()
        IN_FLIGHT_REQUESTS.labels(service=service_name).dec()
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        async with httpx.AsyncClient(base_url=service_url) as client:
            target_url = httpx.URL(request.url.path)
            req = client.build_request(
                request.method,
                target_url,
                headers=request.headers,
                content=await request.body(),
                params=request.query_params,
            )
            resp = await client.send(req, stream=True)

            async def close_response_stream():
                await resp.aclose()

            response = Response(
                content=resp.aiter_bytes(),
                status_code=resp.status_code,
                headers=resp.headers,
                background=BackgroundTask(close_response_stream),
            )
            REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, service=service_name, status_code=resp.status_code).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path, service=service_name).observe(time.time() - start_time)
            return response
    except httpx.RequestError as e:
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, service=service_name, status_code=500).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path, service=service_name).observe(time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Service {service_name} is unavailable: {e}")
    finally:
        IN_FLIGHT_REQUESTS.labels(service=service_name).dec()

@app.api_route("/a02/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def a02_proxy(request: Request, path: str):
    return await _reverse_proxy(request, A02_SERVICE_URL, "a02")

@app.api_route("/a05/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def a05_proxy(request: Request, path: str):
    return await _reverse_proxy(request, A05_SERVICE_URL, "a05")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API Gateway is healthy"}

@app.get("/metrics")
async def get_metrics():
    # This endpoint is handled by Prometheus client's start_http_server
    # We just need to define it to avoid 404 if accessed directly
    return Response(content="# HELP http_requests_total Total HTTP requests\n# TYPE http_requests_total counter\nhttp_requests_total{method=\"GET\",endpoint=\"/metrics\",service=\"gateway\",status_code=\"200\"} 1.0\n", media_type="text/plain")

# Example usage for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


