
filepath = 'tests/test_performance_benchmarks.py'
with open(filepath) as f:
    content = f.read()


search_block = """    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        \"\"\"Simulate batch embedding.\"\"\"
        self.embed_count += len(texts)
        await asyncio.sleep(self._latency_ms / 1000)
        return [await self.embed_text(text) for text in texts]"""

replace_block = """    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        \"\"\"Simulate batch embedding.\"\"\"
        # Note: calling self.embed_text will sleep again and increment count again
        # We need to compute embeddings directly to properly simulate batching
        import hashlib
        self.embed_count += len(texts)
        await asyncio.sleep(self._latency_ms / 1000)
        res = []
        for text in texts:
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            res.append([(hash_val % 1000) / 1000.0] * self._dimension)
        return res"""

new_content = content.replace(search_block, replace_block)
if new_content != content:
    with open(filepath, 'w') as f:
        f.write(new_content)
    print("Patched successfully")
else:
    print("Search block not found")
