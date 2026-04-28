## 2023-10-27 - Vectorized MMR implementation
**Learning:** The MMR (Maximal Marginal Relevance) implementation in `SemanticRouter._apply_mmr` uses a nested loop in pure Python to calculate cosine similarity between the current candidate and all previously selected items (`O(K * |candidates| * |selected|)` dot products).
**Action:** Replace the nested loop with a vectorized approach using numpy. By normalizing the embeddings upfront and calculating similarities via matrix multiplication `np.dot(normed_emb, last_selected_emb)`, we can drastically reduce CPU time. Maintain a running array of `max_sims` to avoid re-evaluating similarities with all previously selected tools at each step.

## 2026-04-16 - Optimize string serialization of embedding vectors
**Learning:** Using Python's built-in `json.dumps()` is significantly faster and safer for serializing large numerical lists (like embeddings) compared to generator expressions with string concatenation (`"[" + ",".join(str(x) for x in embedding) + "]"`), and it is much safer than `str()` because it reliably produces a Postgres array-compatible format (`[1.0, 2.0]`) regardless of whether the input is a list, tuple, or numpy array.
**Action:** Use `json.dumps()` whenever bulk-converting lists to string representations.

## 2023-10-27 - Fast JSON float array serialization
**Learning:** For converting large lists of floats (like vector embeddings) into strings for database queries, `json.dumps(embedding)` is measurably faster (about ~7-8% speedup) and more readable than using a generator expression with string concatenation `("[" + ",".join(str(x) for x in embedding) + "]")`.
**Action:** Use standard `json.dumps` instead of manual string parsing when passing vector data as strings to database drivers like `asyncpg`.

## 2023-10-27 - Fast token matching with substring pre-check
**Learning:** In string parsing functions that use compiled regular expressions (like `_get_token_pattern(token).search(text)`), evaluating the regex engine is expensive. When the token does not exist in the text at all, the regex engine still scans the whole string.
**Action:** Always add a fast substring pre-check (`if token not in text: return False`) before running the regular expression. The native Python `in` operator uses highly optimized C algorithms and provides a massive speedup (>10x in microbenchmarks) for the non-matching case.

## 2023-10-27 - Optimize capability filtering in loops with sets
**Learning:** In `agent_gantry/core/mcp_router.py`, the `filter_by_capabilities` method iterated over servers and evaluated capability constraints using a Python generator expression: `all(cap in server.capabilities for cap in required_capabilities)`. Inside tight loops, this incurs a high performance penalty because the logic is evaluated within the Python interpreter.
**Action:** Always convert constraint lists to sets (`set(required_capabilities)`) before loop execution, and use the native `set.issubset(target)` method. This offloads the logic to optimized C algorithms, yielding a ~3x performance boost according to synthetic benchmarks.

## 2023-10-27 - Set-based filtering optimizations merged
**Learning:** A proposed PR to optimize capability filtering in `mcp_router.py` using `set.issubset()` was closed as a duplicate because this optimization was already merged simultaneously for both `router.py` and `mcp_router.py` in PR #130.
**Action:** When finding optimization opportunities like `all()` generator expressions, I should check whether there are multiple files sharing similar logic (like `router.py` alongside `mcp_router.py`) and ensure I optimize both to avoid duplicated effort across PRs.
