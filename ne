
from langchain.globals import set_llm_cache
from langchain.cache import BaseCache, SQLiteCache  # or RedisCache, GPTCache

class LoggingCache(BaseCache):
    def __init__(self, inner: BaseCache):
        self.inner = inner

    def lookup(self, prompt: str, llm_string: str):
        res = self.inner.lookup(prompt, llm_string)
        print(f"[LLM CACHE {'HIT' if res else 'MISS'}] model={llm_string}")
        return res

    def update(self, prompt: str, llm_string: str, return_val):
        print(f"[LLM CACHE UPDATE] model={llm_string}")
        return self.inner.update(prompt, llm_string, return_val)

# enable once at startup
set_llm_cache(LoggingCache(SQLiteCache(".lc.db")))


# ✅ Works across versions (0.0.x … 0.2+)
try:
    # Newer LangChain (0.2+)
    from langchain_core.globals import set_llm_cache
    from langchain_core.caches import InMemoryCache  # always available
    try:
        from langchain_community.cache import SQLiteCache  # requires langchain-community
        CacheImpl = SQLiteCache
        CACHE_ARGS = { "database_path": ".lc.db" }
    except Exception:
        # Fallback if SQLiteCache not available
        CacheImpl = InMemoryCache
        CACHE_ARGS = {}
except ImportError:
    # Older LangChain (<0.2)
    from langchain.globals import set_llm_cache
    try:
        from langchain.cache import SQLiteCache
        CacheImpl = SQLiteCache
        CACHE_ARGS = { "database_path": ".lc.db" }
    except Exception:
        from langchain.cache import InMemoryCache
        CacheImpl = InMemoryCache
        CACHE_ARGS = {}

# Enable cache once at startup
set_llm_cache(CacheImpl(**CACHE_ARGS))
print(f"Using cache backend: {CacheImpl.__name__}")
