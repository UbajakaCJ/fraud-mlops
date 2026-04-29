"""Structured request/response logging middleware."""
import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        t0 = time.perf_counter()

        response = await call_next(request)

        latency = round((time.perf_counter() - t0) * 1000, 2)
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = str(latency)
        return response
