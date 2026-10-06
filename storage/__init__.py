"""存储层：分片 JSON 存储 + 文件锁 + 跨文档实体注册表。"""

from .lock import FileLock, LockTimeout, lock_path_for
from .sharded import ShardedStore, StoreRegistry, _atomic_write_json, _read_json
from .registry_store import EntityRegistry

__all__ = [
    "FileLock",
    "LockTimeout",
    "lock_path_for",
    "ShardedStore",
    "StoreRegistry",
    "EntityRegistry",
    "_atomic_write_json",
    "_read_json",
]
