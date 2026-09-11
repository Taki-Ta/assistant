import hashlib
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.api.auth import verify_bearer
from interview_ai.db.database import get_db
from interview_ai.db.models import Document


router = APIRouter(prefix="/api/v1/test", tags=["test"])

@router.get('/',dependencies=[Depends(verify_bearer)])
def test_authentication() -> dict[str, str]:
    print('test ')
    return {"message": "authenticated"}


