"""Auth Router"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException
from jose import jwt
from passlib.context import CryptContext
from app.core.config import settings
from app.schemas.schemas import TokenRequest, TokenResponse

router = APIRouter()
# sha256_crypt for broad Python 3.12 compat; swap to bcrypt in production Docker image
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# In production: query DB for user
FAKE_USERS = {
    "admin":  {"password": pwd_context.hash("admin"),  "scopes": ["predict", "admin"]},
    "viewer": {"password": pwd_context.hash("viewer"), "scopes": ["predict"]},
}


@router.post("/token", response_model=TokenResponse, summary="Obtain JWT access token")
async def login(payload: TokenRequest):
    """
    Authenticate and receive a Bearer token.

    Use the token in the `Authorization: Bearer <token>` header for all
    protected endpoints.

    **Demo credentials:** `admin / admin`
    """
    user = FAKE_USERS.get(payload.username)
    if not user or not pwd_context.verify(payload.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode(
        {"sub": payload.username, "scopes": user["scopes"], "exp": exp},
        settings.SECRET_KEY, algorithm=settings.ALGORITHM,
    )
    return TokenResponse(access_token=token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
