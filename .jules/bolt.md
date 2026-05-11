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

## 2024-05-18 - Fast List Filtering with set.issubset
**Learning:** Checking elements inside list comprehensions via `all(...)` or `any(...)` generator expressions creates significant overhead in tight loops (e.g., candidate filtering during semantic routing). The overhead comes from iterating over Python generators instead of native C code.
**Action:** Replace `all(x in y)` and `any(x in y)` with fast native `set` operations like `set.issubset(y)` and `set.isdisjoint(y)`. This yields ~3x speedup. In lists/loops, extract the static side into a `set` *before* the loop.

## 2023-10-27 - Fast regex vs generator expressions for string matching
**Learning:** Using generator expressions like `any(kw in text for kw in keywords)` or `any(ord(c) < 32 for c in text)` is a major performance bottleneck in inner loops due to Python generator overhead.
**Action:** Replace these with pre-compiled regular expressions and `.search()`. This offloads the iteration to optimized C code, yielding >5x speedup for character validation and ~40% speedup for multi-keyword matching. When replacing simple `kw in text` checks, omit word boundaries to preserve the exact substring matching behavior.
