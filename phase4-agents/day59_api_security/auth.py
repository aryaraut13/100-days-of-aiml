# day59_api_security/auth.py
import os
import time
from collections import defaultdict
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Valid API keys (in production: store in database)
VALID_API_KEYS = {
    os.getenv("APP_API_KEY", "dev-key-123"): "developer",
    "admin-key-456": "admin",
}

# Rate limiting: max requests per minute per key
RATE_LIMIT    = 10
rate_limit_db: dict = defaultdict(list)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verify the API key is valid."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Add X-API-Key header."
        )
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )
    return api_key


def check_rate_limit(api_key: str) -> None:
    """Rate limit: max 10 requests per minute per key."""
    now      = time.time()
    minute   = 60
    requests = rate_limit_db[api_key]

    # Remove requests older than 1 minute
    rate_limit_db[api_key] = [t for t in requests if now - t < minute]

    if len(rate_limit_db[api_key]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT} requests/minute."
        )

    rate_limit_db[api_key].append(now)