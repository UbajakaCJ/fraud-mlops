"""Transactions Router — prediction audit log (MongoDB backend)."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.database import get_db
from app.models.transaction import TransactionLog
from app.schemas.schemas import PaginatedTransactions, TransactionRecord

router = APIRouter()


@router.get("/", response_model=PaginatedTransactions, summary="List prediction audit log")
async def list_transactions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    fraud_only: bool = Query(False, description="Filter to fraudulent predictions only"),
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    """
    Returns a paginated audit log of all predictions made by the API.

    Every prediction is persisted asynchronously in the background —
    zero latency impact on the prediction endpoint itself.
    """
    col = db["transaction_logs"]
    query_filter = {"is_fraud": True} if fraud_only else {}

    total = await col.count_documents(query_filter)
    skip = (page - 1) * size
    cursor = col.find(query_filter).sort("created_at", -1).skip(skip).limit(size)
    docs = await cursor.to_list(length=size)

    items = [TransactionRecord(**TransactionLog.from_mongo(d).model_dump()) for d in docs]
    return PaginatedTransactions(total=total, page=page, size=size, items=items)


@router.get("/{transaction_id}", response_model=TransactionRecord,
            summary="Get a single transaction record")
async def get_transaction(
    transaction_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
):
    col = db["transaction_logs"]
    doc = await col.find_one({"transaction_id": transaction_id})
    if not doc:
        from fastapi import HTTPException
        raise HTTPException(404, f"Transaction {transaction_id} not found")
    return TransactionRecord(**TransactionLog.from_mongo(doc).model_dump())
