"""
utils/cache.py
In-memory cache with TTL expiration.
Drop-in replacement for Redis on free tier.
Upgrade to Redis by setting REDIS_URL env var.
"""
import os
import json
import time
import hashlib
from typing import Any, Optional


class Cache:
    """Simple TTL cache. Auto-upgrades to Redis if REDIS_URL is set."""

    def __init__(self):
        self._store: dict = {}
        self._redis = None
        self._try_redis()

    def _try_redis(self):
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                print("[Cache] Connected to Redis")
            except Exception as e:
                print(f"[Cache] Redis failed, using memory cache: {e}")
                self._redis = None

    def _key(self, key: str) -> str:
        return f"rollover:{key}"

    def get(self, key: str) -> Optional[Any]:
        k = self._key(key)
        if self._redis:
            val = self._redis.get(k)
            return json.loads(val) if val else None

        entry = self._store.get(k)
        if not entry:
            return None
        if time.time() > entry["expires"]:
            del self._store[k]
            return None
        return entry["value"]

    def set(self, key: str, value: Any, ttl: int = 1800):
        """Store value with TTL in seconds. Default 30 minutes."""
        k = self._key(key)
        if self._redis:
            self._redis.setex(k, ttl, json.dumps(value))
            return

        self._store[k] = {
            "value":   value,
            "expires": time.time() + ttl,
        }

    def delete(self, key: str):
        k = self._key(key)
        if self._redis:
            self._redis.delete(k)
            return
        self._store.pop(k, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def make_key(self, *args) -> str:
        """Create a hash key from multiple arguments."""
        raw = ":".join(str(a) for a in args)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern (memory cache only)."""
        if self._redis:
            for key in self._redis.scan_iter(f"rollover:{pattern}*"):
                self._redis.delete(key)
            return
        to_delete = [k for k in self._store if pattern in k]
        for k in to_delete:
            del self._store[k]

    def stats(self) -> dict:
        if self._redis:
            info = self._redis.info()
            return {
                "backend": "redis",
                "keys": self._redis.dbsize(),
                "memory_mb": info.get("used_memory_human", "?"),
            }
        active = sum(1 for v in self._store.values() if time.time() <= v["expires"])
        return {"backend": "memory", "keys": active, "total": len(self._store)}


# Cache TTLs
TTL_FIXTURES  = 1800   # 30 min — fixture list
TTL_AI_TIPS   = 1800   # 30 min — AI analysis
TTL_ODDS      = 900    # 15 min — live odds
TTL_STATS     = 3600   # 1 hour — team stats
TTL_ML_PRED   = 1800   # 30 min — ML predictions

# Singleton
_cache: Optional[Cache] = None

def get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache
