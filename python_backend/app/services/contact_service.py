from typing import List

from app.models import Contact
from app.services.wechat_db import WeChatDBReader


def list_contacts() -> List[Contact]:
    return [contact for contact in WeChatDBReader().read_contacts() if not contact.is_group]
