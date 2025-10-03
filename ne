# pip install -U langchain langchain-core langchain-community
from langchain_core.globals import set_llm_cache
from langchain_core.caches import BaseCache
from langchain_community.cache import SQLiteCache  # or RedisCache, etc.

class LoggingCache(BaseCache):
    def __init__(self, inner: BaseCache):
        self.inner = inner

    def lookup(self, prompt: str, llm_string: str):
        res = self.inner.lookup(prompt, llm_string)
        print(f"[LLM CACHE {'HIT' if res else 'MISS'}] {llm_string}")
        return res

    def update(self, prompt: str, llm_string: str, return_val):
        print(f"[LLM CACHE UPDATE] {llm_string}")
        return self.inner.update(prompt, llm_string, return_val)

    def clear(self):
        # delegate if the inner cache supports clear(); otherwise no-op
        if hasattr(self.inner, "clear"):
            return self.inner.clear()
        return None

# enable it
set_llm_cache(LoggingCache(SQLiteCache(".lc.db")))




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



from langchain_core.globals import set_llm_cache
from langchain_core.caches import BaseCache
from langchain_community.cache import GPTCache as LCGPTCache
from langchain_openai import AzureChatOpenAI

# 1) Configure GPTCache storage location (so you can also see files appear)
from gptcache.adapter.api import init_similar_cache  # SQLite + FAISS by default

def init_gptcache(cache_obj, llm_string: str):
    # Writes ./gptcache_sim/sqlite.db and ./gptcache_sim/faiss.index after first write
    init_similar_cache(cache_obj=cache_obj, data_dir="./gptcache_sim")

# 2) Logging wrapper for any LangChain cache backend
class LoggingCache(BaseCache):
    def __init__(self, inner: BaseCache):
        self.inner = inner
    def lookup(self, prompt: str, llm_string: str):
        res = self.inner.lookup(prompt, llm_string)
        print(f"[LLM CACHE {'HIT' if res else 'MISS'}] {llm_string}")
        return res
    def update(self, prompt: str, llm_string: str, return_val):
        print(f"[LLM CACHE UPDATE] stored new entry for {llm_string}")
        return self.inner.update(prompt, llm_string, return_val)
    def clear(self):
        return getattr(self.inner, "clear", lambda: None)()
    def __getattr__(self, name):  # delegate any other attrs
        return getattr(self.inner, name)

# 3) Enable LangChain LLM cache using GPTCache + logging
set_llm_cache(LoggingCache(LCGPTCache(init_gptcache)))
