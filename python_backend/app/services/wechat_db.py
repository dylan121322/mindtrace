import hashlib
import io
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.config import get_settings
from app.models import ChatMessage, Contact
from app.utils.time_utils import unix_to_local_str

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional dependency
    zstd = None


CONTACT_USERNAME_FIELDS = ("username", "user_name", "wxid", "talker", "usr_name")
CONTACT_NICKNAME_FIELDS = ("nickname", "nick_name", "display_name", "conRemark", "name", "chatroom_name", "room_name")
CONTACT_REMARK_FIELDS = ("remark", "con_remark", "remark_name", "alias", "display_remark")
CONTACT_AVATAR_FIELDS = ("small_head_url", "avatar", "avatar_url", "head_img_url", "big_head_url")

MESSAGE_CONTENT_FIELDS = ("message_content", "content", "msg", "text", "str_content", "StrContent")
MESSAGE_TIME_FIELDS = ("create_time", "createTime", "timestamp", "msg_time", "time", "datetime")
MESSAGE_SENDER_FIELDS = ("sender", "from_user", "talker", "user_name", "real_sender")
MESSAGE_SENDER_ID_FIELDS = ("real_sender_id", "sender_id", "from_id")
MESSAGE_IS_MINE_FIELDS = ("is_sender", "is_mine", "from_me", "isSend")
MESSAGE_TYPE_FIELDS = ("local_type", "type", "msg_type")
MESSAGE_CT_FIELDS = ("WCDB_CT_message_content", "compress_type", "content_type")
MESSAGE_SEQ_FIELDS = ("seq", "msg_id", "msgId", "local_id", "id")


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _connect_readonly(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except Exception:
        conn = sqlite3.connect(str(path), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(row[0]) for row in rows]


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(table)})").fetchall()
    return [str(row["name"]) for row in rows]


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for candidate in candidates:
        needle = candidate.lower()
        for col in columns:
            if needle in col.lower():
                return col
    return None


def _select_expr(column: Optional[str], alias: str, default: Any = "") -> str:
    if column:
        return f"{_quote_ident(column)} AS {_quote_ident(alias)}"
    if isinstance(default, (int, float)):
        return f"{default} AS {_quote_ident(alias)}"
    return f"'' AS {_quote_ident(alias)}"


def _normalize_key(target_key: str) -> str:
    key = (target_key or "").strip()
    for prefix in ("contact:", "group:", "all:"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def message_table_name(username_or_table: str) -> str:
    key = _normalize_key(username_or_table)
    if key.startswith("Msg_"):
        return key
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"Msg_{digest}"


def _decompress_zstd(raw: bytes) -> Optional[bytes]:
    if zstd is None:
        return None
    try:
        return zstd.ZstdDecompressor().decompress(raw)
    except Exception:
        try:
            with zstd.ZstdDecompressor().stream_reader(io.BytesIO(raw)) as reader:
                return reader.read()
        except Exception:
            return None


def _looks_like_zstd(raw: bytes, compress_type: Any = None) -> bool:
    if raw.startswith(b"\x28\xb5\x2f\xfd"):
        return True
    try:
        return int(compress_type or 0) == 4
    except (TypeError, ValueError):
        return False


def _decode_content(raw: Any, compress_type: Any = None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        if _looks_like_zstd(raw, compress_type):
            decoded = _decompress_zstd(raw)
            if decoded is not None:
                return decoded.decode("utf-8", errors="ignore")
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _prefer_text(current: str, candidate: str) -> str:
    current = str(current or "").strip()
    candidate = str(candidate or "").strip()
    return current or candidate


def _truthy_db_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().lower()
    if text in ("", "0", "false", "no", "none", "null"):
        return False
    return True


def _merge_contact(existing: Contact, candidate: Contact) -> Contact:
    return Contact(
        username=existing.username or candidate.username,
        nickname=_prefer_text(existing.nickname, candidate.nickname),
        remark=_prefer_text(existing.remark, candidate.remark),
        avatar=_prefer_text(existing.avatar, candidate.avatar),
        is_group=existing.is_group or candidate.is_group,
    )


def _typed_content(local_type: int, content: str) -> str:
    if local_type in (0, 1):
        return content.strip()
    if local_type == 3:
        return "[图片]"
    if local_type == 34:
        return "[语音]"
    if local_type == 43:
        return "[视频]"
    if local_type == 47:
        return "[动画表情]"
    if local_type == 49:
        lowered = content.lower()
        if "wcpay" in lowered or "redenvelope" in lowered:
            return "[红包/转账]"
        if "weappinfo" in lowered or "miniprogram" in lowered:
            return "[小程序]"
        return "[链接/文件]"
    if local_type in (10000, 11000):
        return "[系统消息]"
    return f"[消息类型 {local_type}]"


class WeChatDBReader:
    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else get_settings().data_dir
        self._self_rowid_cache: Dict[str, Optional[int]] = {}

    @property
    def contact_db_path(self) -> Path:
        return self.data_dir / "contact" / "contact.db"

    @property
    def message_dir(self) -> Path:
        return self.data_dir / "message"

    def _message_db_paths(self) -> List[Path]:
        if not self.message_dir.exists():
            return []
        paths = []
        for path in sorted(self.message_dir.glob("message_*.db")):
            name = path.name.lower()
            if "fts" in name or "resource" in name:
                continue
            paths.append(path)
        return paths

    def read_contacts(self) -> List[Contact]:
        conn = _connect_readonly(self.contact_db_path)
        if conn is None:
            return []
        contacts: Dict[str, Contact] = {}
        try:
            for table in _list_tables(conn):
                columns = _table_columns(conn, table)
                username_col = _find_column(columns, CONTACT_USERNAME_FIELDS)
                if not username_col:
                    continue
                nickname_col = _find_column(columns, CONTACT_NICKNAME_FIELDS)
                remark_col = _find_column(columns, CONTACT_REMARK_FIELDS)
                avatar_col = _find_column(columns, CONTACT_AVATAR_FIELDS)
                exprs = [
                    _select_expr(username_col, "username"),
                    _select_expr(nickname_col, "nickname"),
                    _select_expr(remark_col, "remark"),
                    _select_expr(avatar_col, "avatar"),
                ]
                where = ""
                if _find_column(columns, ("verify_flag",)):
                    where = f" WHERE COALESCE({_quote_ident(_find_column(columns, ('verify_flag',)))}, 0) = 0"
                sql = f"SELECT {', '.join(exprs)} FROM {_quote_ident(table)}{where}"
                try:
                    rows = conn.execute(sql).fetchall()
                except sqlite3.Error:
                    continue
                for row in rows:
                    username = str(row["username"] or "").strip()
                    if not username:
                        continue
                    candidate = Contact(
                        username=username,
                        nickname=str(row["nickname"] or ""),
                        remark=str(row["remark"] or ""),
                        avatar=str(row["avatar"] or ""),
                        is_group=username.endswith("@chatroom") or "chatroom" in table.lower(),
                    )
                    if username in contacts:
                        contacts[username] = _merge_contact(contacts[username], candidate)
                    else:
                        contacts[username] = candidate
        finally:
            conn.close()
        return list(contacts.values())

    def read_message_targets(self) -> List[Dict[str, Any]]:
        contacts = self.read_contacts()
        table_stats = self._message_table_stats()
        targets: List[Dict[str, Any]] = []
        for contact in contacts:
            if contact.is_group:
                continue
            table = message_table_name(contact.username)
            stats = table_stats.get(table)
            if not stats:
                continue
            targets.append(
                {
                    "username": contact.username,
                    "nickname": contact.nickname,
                    "remark": contact.remark,
                    "avatar": contact.avatar,
                    "is_group": contact.is_group,
                    "target_table": table,
                    "message_count": stats["message_count"],
                    "first_message_ts": stats["first_message_ts"],
                    "last_message_ts": stats["last_message_ts"],
                    "first_message_time": unix_to_local_str(stats["first_message_ts"]) if stats["first_message_ts"] else "",
                    "last_message_time": unix_to_local_str(stats["last_message_ts"]) if stats["last_message_ts"] else "",
                    "db_count": stats["db_count"],
                }
            )
        targets.sort(key=lambda item: (int(item.get("last_message_ts") or 0), int(item.get("message_count") or 0)), reverse=True)
        return targets

    def _message_table_stats(self) -> Dict[str, Dict[str, int]]:
        stats: Dict[str, Dict[str, int]] = {}
        for path in self._message_db_paths():
            conn = _connect_readonly(path)
            if conn is None:
                continue
            try:
                for table in [t for t in _list_tables(conn) if t.startswith("Msg_")]:
                    columns = _table_columns(conn, table)
                    time_col = _find_column(columns, MESSAGE_TIME_FIELDS)
                    try:
                        if time_col:
                            row = conn.execute(
                                f"SELECT COUNT(*), MIN({_quote_ident(time_col)}), MAX({_quote_ident(time_col)}) FROM {_quote_ident(table)}"
                            ).fetchone()
                        else:
                            row = conn.execute(f"SELECT COUNT(*), 0, 0 FROM {_quote_ident(table)}").fetchone()
                    except sqlite3.Error:
                        continue
                    count = int(row[0] or 0)
                    first_ts = int(row[1] or 0)
                    last_ts = int(row[2] or 0)
                    item = stats.setdefault(
                        table,
                        {"message_count": 0, "first_message_ts": 0, "last_message_ts": 0, "db_count": 0},
                    )
                    item["message_count"] += count
                    item["db_count"] += 1
                    if first_ts and (not item["first_message_ts"] or first_ts < item["first_message_ts"]):
                        item["first_message_ts"] = first_ts
                    if last_ts and last_ts > item["last_message_ts"]:
                        item["last_message_ts"] = last_ts
            finally:
                conn.close()
        return stats

    def _contact_name_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for contact in self.read_contacts():
            display = contact.remark or contact.nickname or contact.username
            mapping[contact.username] = display
        return mapping

    def _name2id_map(self, conn: sqlite3.Connection) -> Dict[int, str]:
        tables = set(_list_tables(conn))
        if "Name2Id" not in tables:
            return {}
        try:
            rows = conn.execute("SELECT rowid, user_name FROM Name2Id").fetchall()
            return {int(row[0]): str(row[1]) for row in rows}
        except sqlite3.Error:
            return {}

    def _contact_rowid(self, conn: sqlite3.Connection, username: str) -> Optional[int]:
        if not username:
            return None
        try:
            row = conn.execute("SELECT rowid FROM Name2Id WHERE user_name = ? LIMIT 1", (username,)).fetchone()
        except sqlite3.Error:
            return None
        return int(row[0]) if row else None

    def _conn_cache_key(self, conn: sqlite3.Connection) -> str:
        try:
            row = conn.execute("PRAGMA database_list").fetchone()
            return str(row[2] or id(conn)) if row else str(id(conn))
        except sqlite3.Error:
            return str(id(conn))

    def _infer_self_rowid(self, conn: sqlite3.Connection) -> Optional[int]:
        cache_key = self._conn_cache_key(conn)
        if cache_key in self._self_rowid_cache:
            return self._self_rowid_cache[cache_key]

        tables = set(_list_tables(conn))
        if "Name2Id" not in tables:
            self._self_rowid_cache[cache_key] = None
            return None

        sender_counts: Dict[int, int] = {}
        inspected = 0
        for contact in self.read_contacts():
            if contact.is_group:
                continue
            table = message_table_name(contact.username)
            if table not in tables:
                continue
            contact_rowid = self._contact_rowid(conn, contact.username)
            if not contact_rowid:
                continue
            columns = _table_columns(conn, table)
            sender_id_col = _find_column(columns, MESSAGE_SENDER_ID_FIELDS)
            if not sender_id_col:
                continue
            try:
                rows = conn.execute(
                    f"SELECT {_quote_ident(sender_id_col)} AS sender_id "
                    f"FROM (SELECT {_quote_ident(sender_id_col)} FROM {_quote_ident(table)} ORDER BY rowid DESC LIMIT 500)"
                ).fetchall()
            except sqlite3.Error:
                continue
            for row in rows:
                try:
                    sender_id = int(row["sender_id"] or 0)
                except (TypeError, ValueError):
                    sender_id = 0
                if sender_id and sender_id != contact_rowid:
                    sender_counts[sender_id] = sender_counts.get(sender_id, 0) + 1
            inspected += 1
            if inspected >= 80 and sender_counts:
                break

        if not sender_counts:
            self._self_rowid_cache[cache_key] = None
            return None
        self_rowid, count = max(sender_counts.items(), key=lambda item: item[1])
        result = self_rowid if count >= 3 else None
        self._self_rowid_cache[cache_key] = result
        return result

    def read_self_usernames(self) -> List[str]:
        usernames: List[str] = []
        seen = set()
        for path in self._message_db_paths():
            conn = _connect_readonly(path)
            if conn is None:
                continue
            try:
                self_rowid = self._infer_self_rowid(conn)
                if self_rowid is None:
                    continue
                username = self._name2id_map(conn).get(self_rowid, "")
                if username and username not in seen:
                    seen.add(username)
                    usernames.append(username)
            finally:
                conn.close()
        return usernames

    def read_messages(
        self,
        target_key: str,
        target_type: str = "contact",
        time_from: Optional[int] = None,
        time_to: Optional[int] = None,
        limit: int = 500,
    ) -> List[ChatMessage]:
        messages, _ = self.read_messages_with_diagnostics(target_key, target_type, time_from, time_to, limit)
        return messages

    def read_messages_with_diagnostics(
        self,
        target_key: str,
        target_type: str = "contact",
        time_from: Optional[int] = None,
        time_to: Optional[int] = None,
        limit: int = 500,
    ) -> Tuple[List[ChatMessage], Dict[str, Any]]:
        key = _normalize_key(target_key or "")
        diag: Dict[str, Any] = {
            "target_type": target_type,
            "target_table": message_table_name(key) if key and target_type not in ("all", "self") else "",
            "message_db_count": 0,
            "candidate_table_count": 0,
            "target_table_found": False,
            "target_table_total_rows": 0,
            "time_range_rows": 0,
            "compressed_rows": 0,
            "decoded_message_count": 0,
            "time_from": time_from,
            "time_to": time_to,
            "zstd_available": zstd is not None,
        }
        if target_type == "group":
            diag["reason"] = "group_analysis_disabled"
            return [], diag
        if not key and target_type not in ("all", "self"):
            diag["reason"] = "empty_target_key"
            return [], diag
        target_table = message_table_name(key) if target_type not in ("all", "self") else ""
        name_map = self._contact_name_map()
        allowed_tables = {message_table_name(contact.username) for contact in self.read_contacts() if not contact.is_group}
        all_messages: List[ChatMessage] = []
        paths = self._message_db_paths()
        diag["message_db_count"] = len(paths)
        for path in paths:
            conn = _connect_readonly(path)
            if conn is None:
                continue
            try:
                tables = _list_tables(conn)
                self_rowid = self._infer_self_rowid(conn) if target_type in ("all", "self") else None
                if self_rowid is not None:
                    diag["self_sender_inferred"] = True
                if target_type in ("all", "self"):
                    candidate_tables = [t for t in tables if t.startswith("Msg_") and t in allowed_tables]
                elif target_table in tables:
                    candidate_tables = [target_table]
                elif key.startswith("Msg_") and key in tables:
                    candidate_tables = [key]
                else:
                    candidate_tables = []
                diag["candidate_table_count"] += len(candidate_tables)
                if target_table and target_table in candidate_tables:
                    diag["target_table_found"] = True
                for table in candidate_tables:
                    table_diag = self._table_message_counts(conn, table, time_from, time_to)
                    diag["target_table_total_rows"] += table_diag.get("total_rows", 0)
                    diag["time_range_rows"] += table_diag.get("time_range_rows", 0)
                    diag["compressed_rows"] += table_diag.get("compressed_rows", 0)
                    all_messages.extend(
                        self._read_table_messages(
                            conn,
                            table,
                            key or table,
                            target_type,
                            time_from,
                            time_to,
                            name_map,
                            self_rowid,
                        )
                    )
            finally:
                conn.close()
        all_messages.sort(key=lambda m: (m.datetime, m.seq))
        if limit and limit > 0 and len(all_messages) > limit:
            all_messages = all_messages[-limit:]
        diag["decoded_message_count"] = len(all_messages)
        if not all_messages:
            if diag["candidate_table_count"] == 0:
                diag["reason"] = "target_message_table_not_found"
            elif diag["time_range_rows"] == 0:
                diag["reason"] = "no_rows_in_selected_time_range"
            else:
                diag["reason"] = "rows_found_but_no_decodable_messages"
        return all_messages, diag

    def _table_message_counts(
        self,
        conn: sqlite3.Connection,
        table: str,
        time_from: Optional[int],
        time_to: Optional[int],
    ) -> Dict[str, int]:
        columns = _table_columns(conn, table)
        time_col = _find_column(columns, MESSAGE_TIME_FIELDS)
        ct_col = _find_column(columns, MESSAGE_CT_FIELDS)
        out = {"total_rows": 0, "time_range_rows": 0, "compressed_rows": 0}
        try:
            out["total_rows"] = int(conn.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}").fetchone()[0] or 0)
        except sqlite3.Error:
            return out
        where: List[str] = []
        params: List[Any] = []
        if time_col and time_from is not None:
            where.append(f"{_quote_ident(time_col)} >= ?")
            params.append(time_from)
        if time_col and time_to is not None:
            where.append(f"{_quote_ident(time_col)} <= ?")
            params.append(time_to)
        if not where:
            out["time_range_rows"] = out["total_rows"]
            if ct_col:
                try:
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE COALESCE({_quote_ident(ct_col)}, 0) = 4"
                    ).fetchone()
                    out["compressed_rows"] = int(row[0] or 0)
                except sqlite3.Error:
                    out["compressed_rows"] = 0
            return out
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
            out["time_range_rows"] = int(row[0] or 0)
        except sqlite3.Error:
            out["time_range_rows"] = 0
        if ct_col:
            compressed_where = list(where)
            compressed_params = list(params)
            compressed_where.append(f"COALESCE({_quote_ident(ct_col)}, 0) = 4")
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE {' AND '.join(compressed_where)}",
                    compressed_params,
                ).fetchone()
                out["compressed_rows"] = int(row[0] or 0)
            except sqlite3.Error:
                out["compressed_rows"] = 0
        return out

    def _read_table_messages(
        self,
        conn: sqlite3.Connection,
        table: str,
        contact_key: str,
        target_type: str,
        time_from: Optional[int],
        time_to: Optional[int],
        name_map: Dict[str, str],
        self_rowid: Optional[int] = None,
    ) -> List[ChatMessage]:
        columns = _table_columns(conn, table)
        content_col = _find_column(columns, MESSAGE_CONTENT_FIELDS)
        if not content_col:
            return []
        time_col = _find_column(columns, MESSAGE_TIME_FIELDS)
        sender_col = _find_column(columns, MESSAGE_SENDER_FIELDS)
        sender_id_col = _find_column(columns, MESSAGE_SENDER_ID_FIELDS)
        is_mine_col = _find_column(columns, MESSAGE_IS_MINE_FIELDS)
        type_col = _find_column(columns, MESSAGE_TYPE_FIELDS)
        ct_col = _find_column(columns, MESSAGE_CT_FIELDS)
        seq_col = _find_column(columns, MESSAGE_SEQ_FIELDS)
        exprs = [
            "rowid AS _rowid",
            _select_expr(seq_col, "seq", 0),
            _select_expr(time_col, "create_time", 0),
            _select_expr(content_col, "content"),
            _select_expr(sender_col, "sender"),
            _select_expr(sender_id_col, "sender_id", 0),
            _select_expr(is_mine_col, "is_mine", 0),
            _select_expr(type_col, "local_type", 1),
            _select_expr(ct_col, "compress_type", 0),
        ]
        where: List[str] = []
        params: List[Any] = []
        if time_col and time_from is not None:
            where.append(f"{_quote_ident(time_col)} >= ?")
            params.append(time_from)
        if time_col and time_to is not None:
            where.append(f"{_quote_ident(time_col)} <= ?")
            params.append(time_to)
        where_sql = " WHERE " + " AND ".join(where) if where else ""
        order_sql = f" ORDER BY {_quote_ident(time_col)} ASC" if time_col else " ORDER BY rowid ASC"
        sql = f"SELECT {', '.join(exprs)} FROM {_quote_ident(table)}{where_sql}{order_sql}"
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []

        id_to_wxid = self._name2id_map(conn)
        contact_rowid = self._contact_rowid(conn, contact_key)
        messages: List[ChatMessage] = []
        for row in rows:
            local_type = int(row["local_type"] or 1)
            raw_text = _decode_content(row["content"], row["compress_type"]).strip()
            speaker_wxid = ""
            content = raw_text
            if local_type == 1 and ":\n" in raw_text:
                prefix, body = raw_text.split(":\n", 1)
                if 0 < len(prefix) < 80:
                    speaker_wxid = prefix.strip()
                    content = body.strip()
            content = _typed_content(local_type, content)
            if not content:
                continue
            sender_id = int(row["sender_id"] or 0)
            if not speaker_wxid and sender_id in id_to_wxid:
                speaker_wxid = id_to_wxid[sender_id]
            is_mine = False
            if is_mine_col:
                is_mine = _truthy_db_value(row["is_mine"])
            if self_rowid is not None and sender_id:
                is_mine = is_mine or sender_id == self_rowid
            elif target_type == "contact" and contact_rowid is not None and sender_id:
                is_mine = sender_id != contact_rowid
            elif str(row["sender"] or "").lower() in ("me", "self", "mine", "我"):
                is_mine = True

            if target_type == "self" and not is_mine:
                continue

            if target_type == "group":
                sender = "我" if is_mine else name_map.get(speaker_wxid, speaker_wxid or str(row["sender"] or "未知"))
            elif is_mine:
                sender = "我"
            else:
                sender = "对方"
            ts = int(row["create_time"] or 0)
            seq_value = int(row["seq"] or row["_rowid"] or 0)
            messages.append(
                ChatMessage(
                    seq=seq_value,
                    datetime=unix_to_local_str(ts) if ts else str(row["create_time"] or ""),
                    sender=sender,
                    content=content,
                    is_mine=is_mine,
                    contact_key=contact_key,
                )
            )
        return messages
