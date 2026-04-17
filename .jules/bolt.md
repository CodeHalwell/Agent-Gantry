## 2023-10-27 - Vectorized MMR implementation
**Learning:** The MMR (Maximal Marginal Relevance) implementation in `SemanticRouter._apply_mmr` uses a nested loop in pure Python to calculate cosine similarity between the current candidate and all previously selected items (`O(K * |candidates| * |selected|)` dot products).
**Action:** Replace the nested loop with a vectorized approach using numpy. By normalizing the embeddings upfront and calculating similarities via matrix multiplication `np.dot(normed_emb, last_selected_emb)`, we can drastically reduce CPU time. Maintain a running array of `max_sims` to avoid re-evaluating similarities with all previously selected tools at each step.

## 2023-10-27 - Fast JSON float array serialization
**Learning:** For converting large lists of floats (like vector embeddings) into strings for database queries, `json.dumps(embedding)` is measurably faster (about ~7-8% speedup) and more readable than using a generator expression with string concatenation `("[" + ",".join(str(x) for x in embedding) + "]")`.
**Action:** Use standard `json.dumps` instead of manual string parsing when passing vector data as strings to database drivers like `asyncpg`.
