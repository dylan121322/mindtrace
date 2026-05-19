from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.env_config_service import public_env, save_env_file


router = APIRouter(tags=["python-config"])


class PythonConfigUpdate(BaseModel):
    values: Dict[str, str] = Field(default_factory=dict)


@router.get("/python-config")
def get_python_config() -> Dict:
    return public_env()


@router.put("/python-config")
def put_python_config(body: PythonConfigUpdate) -> Dict:
    return save_env_file(body.values)

