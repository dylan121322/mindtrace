from typing import List

from fastapi import APIRouter

from app.models import Contact
from app.services.contact_service import list_contacts


router = APIRouter(tags=["contacts"])


@router.get("/contacts", response_model=List[Contact])
def get_contacts() -> List[Contact]:
    return list_contacts()

