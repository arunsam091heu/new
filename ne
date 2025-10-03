
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
