"""Transactions Router — prediction audit log."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.core.database import get_db
from app.models.transaction import TransactionLog
from app.schemas.schemas import PaginatedTransactions, TransactionRecord

router = APIRouter()


@router.get("/", response_model=PaginatedTransactions, summary="List prediction audit log")
async def list_transactions(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    fraud_only: bool = Query(False, description="Filter to fraudulent predictions only"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a paginated audit log of all predictions made by the API.

    Every prediction is persisted asynchronously in the background —
    zero latency impact on the prediction endpoint itself.
    """
    stmt = select(TransactionLog)
    if fraud_only:
        stmt = stmt.where(TransactionLog.is_fraud == True)
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar()
    stmt = stmt.order_by(TransactionLog.created_at.desc()).offset((page-1)*size).limit(size)
    rows = (await db.execute(stmt)).scalars().all()
    return PaginatedTransactions(
        total=total, page=page, size=size,
        items=[TransactionRecord.model_validate(r) for r in rows],
    )


@router.get("/{transaction_id}", response_model=TransactionRecord,
            summary="Get a single transaction record")
async def get_transaction(transaction_id: str, db: AsyncSession = Depends(get_db)):
    stmt = select(TransactionLog).where(TransactionLog.transaction_id == transaction_id)
    row = (await db.execute(stmt)).scalar_one_or_none()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, f"Transaction {transaction_id} not found")
    return TransactionRecord.model_validate(row)
